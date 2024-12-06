import streamlit as st
import os
import json
from datetime import datetime
from typing import Dict, Tuple
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
import anthropic

class ContentProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.anthropic_client = anthropic.Client(api_key=api_key)

    def extract_labels(self, text: str, existing_labels: list = None, prompt_template: str = None) -> list:
        """Extract labels from a chunk of text, considering existing labels."""
        context = ""
        if existing_labels:
            context = ", ".join(existing_labels)
        
        # Replace placeholders in prompt template
        prompt = prompt_template.replace("{text}", text).replace("{existing_labels}", context)
        
        # Using completion() instead of messages.create()
        response = self.anthropic_client.completion(
            prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
            model="claude-2.1",
            max_tokens_to_sample=200,
            temperature=0.3,
        )
        
        # Extract labels from response
        new_labels = [label.strip() for label in response.completion.strip().split('\n')]
        print(f"Extracted labels: {new_labels}")
        return new_labels

    def update_label_set(self, current_labels: list, new_labels: list) -> list:
        """Update the set of labels, maintaining unique entries."""
        # Convert to set to remove duplicates while preserving order
        updated = list(dict.fromkeys(current_labels + new_labels))
        return updated[:20]  # Keep top 20 labels to avoid context getting too long
    
    def chunk_text(self, text: str, chunk_max_size: int = 2000, overlap_percentage: float = 0.10) -> list:
        overlap_size = int(chunk_max_size * overlap_percentage)
        chunk_size = chunk_max_size - overlap_size
        num_chunks = int((len(text) + chunk_size - 1) / chunk_size)
        chunks = []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size + overlap_size, len(text))
            chunks.append(text[start:end])
        return chunks
    
    def summarize_chunk(self, text: str, summary_prompt: str, label_prompt: str, level=1) -> Tuple[str, Dict[str, Dict[str, str]]]:
        texts = self.chunk_text(text)
        summaries = {}
        current_labels = []
        
        print(f"\n=== Starting Level {level} ===")
        print(f"Number of chunks to process: {len(texts)}")
        
        # Create placeholders for UI elements
        level_container = st.empty()
        progress_container = st.empty()
        labels_container = st.empty()
        
        with level_container.container():
            st.write(f"### Level {level} Summaries")
        
        with progress_container.container():
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        current_summaries = []
        for i, text_ in enumerate(texts):
            print(f"\nProcessing chunk {i+1}/{len(texts)} at level {level}")
            status_text.write(f'Processing chunk {i+1} of {len(texts)}...')
            
            # Extract and update labels
            chunk_labels = self.extract_labels(text_, current_labels, label_prompt)
            current_labels = self.update_label_set(current_labels, chunk_labels)
            
            # Display current labels
            with labels_container.container():
                st.write("### Current Topics")
                st.write(", ".join(current_labels))
            
            # Include labels in summarization context
            labels_context = f"Current topic labels: {', '.join(current_labels)}\n\n"
            prompt = labels_context + prompt_template.replace("{text}", text_)
            
            # Using completion() instead of messages.create()
            response = self.anthropic_client.completion(
                prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                model="claude-2.1",
                max_tokens_to_sample=1000,
                temperature=0.7,
            )
            summary = response.completion.strip()
            current_summaries.append(summary)
            
            key = f"Level {level} - Chunk {i+1}"
            summaries[key] = {
                "original": text_,
                "summary": summary,
                "labels": chunk_labels,
                "accumulated_labels": current_labels.copy()
            }
            
            # Update the progress bar
            progress_bar.progress((i + 1) / len(texts))
            
            # Update the display for this chunk
            with level_container.container():
                st.write(f"### Level {level} Summaries")
                for j in range(i + 1):
                    chunk_key = f"Level {level} - Chunk {j+1}"
                    with st.expander(f"Show {chunk_key}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original Text:**")
                            st.write(summaries[chunk_key]["original"])
                            st.markdown("**Chunk Labels:**")
                            st.write(", ".join(summaries[chunk_key]["labels"]))
                        with col2:
                            st.markdown("**Summary:**")
                            st.write(summaries[chunk_key]["summary"])
                            st.markdown("**Accumulated Labels:**")
                            st.write(", ".join(summaries[chunk_key]["accumulated_labels"]))
        
        joined_summaries = '\n'.join(current_summaries)
        progress_container.empty()
        
        if len(joined_summaries) < 2000:
            summaries["Final Summary"] = {
                "original": text,
                "summary": joined_summaries,
                "labels": current_labels,
                "accumulated_labels": current_labels
            }
            return joined_summaries, summaries
        
        level += 1
        final_summary, recursive_summaries = self.summarize_chunk(joined_summaries, summary_prompt, label_prompt, level=level)
        
        # Update summaries with recursive results
        for k, v in recursive_summaries.items():
            if k != "Final Summary":
                new_key = f"Level {level} - {k.split(' - ', 1)[-1]}"
                summaries[new_key] = v
        
        summaries["Final Summary"] = {
            "original": text,
            "summary": final_summary,
            "labels": current_labels,
            "accumulated_labels": current_labels
        }
        return final_summary, summaries
    
    def process_youtube_transcript(self, video_id: str, summary_prompt: str, label_prompt: str) -> Tuple[str, str, Dict[str, Dict[str, str]]]:
        with st.spinner('Fetching YouTube transcript...'):
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([segment["text"] for segment in transcript_list])
        
        summary, summaries_dict = self.summarize_chunk(transcript_text, summary_prompt, label_prompt)
        return transcript_text, summary, summaries_dict

    def process_website(self, url: str, summary_prompt: str, label_prompt: str) -> Tuple[str, str, Dict[str, Dict[str, str]]]:
        with st.spinner('Fetching website content...'):
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
        
        summary, summaries_dict = self.summarize_chunk(text, summary_prompt, label_prompt)
        return text, summary, summaries_dict

def export_summaries(summaries: Dict[str, Dict[str, str]], content_type: str, source: str) -> Tuple[str, bytes]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"summary_{content_type}_{timestamp}.json"
    
    export_data = {
        "content_type": content_type,
        "source": source,
        "timestamp": timestamp,
        "summaries": summaries
    }
    
    # Convert to JSON string
    json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
    
    return filename, json_str.encode('utf-8')

def main():
    st.title("Content Summarizer")
    
    if 'current_summaries' not in st.session_state:
        st.session_state.current_summaries = None
    if 'current_source' not in st.session_state:
        st.session_state.current_source = None
    if 'current_type' not in st.session_state:
        st.session_state.current_type = None
    
    api_key = st.text_input("Enter your Anthropic API Key:", type="password")

    # Add custom prompt templates
    default_summary_prompt = "You are part of a summarisation subsystem. Your job is to summarise the information given to you. The information may already be a summary of larger sections of text. Please summarize the following text:\n{text}"
    default_label_prompt = """Based on the following text, generate exactly 5 key topic labels. 
    If any of the themes match existing labels, reuse those exact labels.
    
    existing labels: {existing_labels}
    Text: {text}
    Return ONLY the labels, one per line, without any other text before or after the labels

    example response: label_1\nlabel_2\nlabel_3\nlabel_4\nlabel_5"""

    prompt_template = st.text_area(
        "Summarization Prompt Template",
        value=default_summary_prompt,
        help="Use {text} as a placeholder for the content to be summarized"
    )

    label_prompt_template = st.text_area(
        "Label Extraction Prompt Template",
        value=default_label_prompt,
        help="Use {text} as placeholder for content and {existing_labels} for current labels"
    )
    
    if not api_key:
        st.warning("Please enter your API key to continue.")
        return
    
    processor = ContentProcessor(api_key)
    
    input_mode = st.radio(
        "Select input mode:",
        ["YouTube Video", "Website URL", "Text Input"]
    )
    
    if input_mode == "YouTube Video":
        video_url = st.text_input("Enter YouTube Video URL:")
        if video_url and st.button("Process Video"):
            try:
                video_id = video_url.split("v=")[1]
                text, summary, summaries = processor.process_youtube_transcript(video_id, prompt_template)
                st.session_state.current_summaries = summaries
                st.session_state.current_source = video_url
                st.session_state.current_type = "youtube"
                st.success("Video processed successfully!")
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
    
    elif input_mode == "Website URL":
        url = st.text_input("Enter Website URL:")
        if url and st.button("Process Website"):
            try:
                text, summary, summaries = processor.process_website(url, prompt_template)
                st.session_state.current_summaries = summaries
                st.session_state.current_source = url
                st.session_state.current_type = "website"
                st.success("Website processed successfully!")
            except Exception as e:
                st.error(f"Error processing website: {str(e)}")
    
    else:  # Text Input
        text = st.text_area("Enter your text:")
        if text and st.button("Process Text"):
            try:
                summary, summaries = processor.summarize_chunk(text, prompt_template, label_prompt_template)
                st.session_state.current_summaries = summaries
                st.session_state.current_source = "direct_input"
                st.session_state.current_type = "text"
                st.success("Text processed successfully!")
            except Exception as e:
                st.error(f"Error processing text: {str(e)}")
    
    # Display summaries if available
    if st.session_state.current_summaries:
        st.subheader("Content Analysis Results")
        
        # Show final summary first
        st.write("### Final Summary")
        final_summary_data = st.session_state.current_summaries["Final Summary"]
        st.write(final_summary_data["summary"])
        st.write("### Final Topics")
        st.write(", ".join(final_summary_data["accumulated_labels"]))
        
        with st.expander("Show original full text"):
            st.write(final_summary_data["original"])
        
        # Show intermediate summaries in expandable sections
        st.write("### Intermediate Summaries")
        sorted_keys = sorted([k for k in st.session_state.current_summaries.keys() if k != "Final Summary"])
        
        for key in sorted_keys:
            with st.expander(f"Show {key}"):
                chunk_data = st.session_state.current_summaries[key]
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Text:**")
                    st.write(chunk_data["original"])
                with col2:
                    st.markdown("**Summary:**")
                    st.write(chunk_data["summary"])
        
        # Add export button
        if st.button("Export Summaries"):            
            # Create download button for export
            filename, file_content = export_summaries(
                st.session_state.current_summaries,
                st.session_state.current_type,
                st.session_state.current_source
            )
            
            st.download_button(
                label="Download Summaries",
                data=file_content,
                file_name=filename,
                mime="application/json"
            )

if __name__ == "__main__":
    main()