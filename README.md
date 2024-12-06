# Content Summarizer

A Streamlit application that summarizes content from various sources (YouTube videos, websites, or text input) using the Anthropic Claude API.

## Features

- Multiple input sources support (YouTube, websites, direct text)
- Recursive summarization with intermediate results
- Dynamic topic labeling
- Customizable summarization and labeling prompts
- Export functionality
- Real-time progress tracking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/content-summarizer.git
cd content-summarizer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run src/app.py
```

2. Enter your Anthropic API key when prompted

3. Choose your input source:
   - YouTube video URL
   - Website URL
   - Direct text input

4. Customize the summarization and labeling prompts if desired

5. Process your content and view the results

6. Download the results as JSON if needed

## Configuration

You can customize the Streamlit appearance and behavior by modifying `.streamlit/config.toml`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)