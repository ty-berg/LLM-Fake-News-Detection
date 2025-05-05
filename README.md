
# LLM-Fake-News-Detection

## Overview

This project explores the capabilities of advanced Large Language Models (LLMs) in detecting fake news. By leveraging models like GPT-4 and LLaMA, it aims to assess their effectiveness in identifying misinformation within news articles.

## Repository Contents

- **`test-gpt4.py`**: Script to evaluate fake news detection using OpenAI's GPT-4 model.
- **`test-llama.py`**: Script to assess fake news detection using Meta's LLaMA model.
- **`refined_cleaned_data.jsonl`**: A preprocessed dataset in JSON Lines format, containing news articles labeled for authenticity.
- **`Fake_News_Detection_With_Advanced_LLMs.pdf`**: A detailed report outlining the methodology, experiments, and findings of the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages:

```bash
pip install openai transformers
```

### Setup

1. **Clone the Repository**:

```bash
git clone https://github.com/ty-berg/LLM-Fake-News-Detection.git
cd LLM-Fake-News-Detection
```

2. **Configure API Access**:

- For GPT-4:
  - Obtain an API key from [OpenAI](https://platform.openai.com/).
  - Set the API key as an environment variable:
    ```bash
    export OPENAI_API_KEY='your-api-key-here'
    ```

- For LLaMA:
  - Ensure you have access to the LLaMA model weights and necessary configurations.
  - Refer to Meta's official [LLaMA repository](https://github.com/facebookresearch/llama) for setup instructions.

## Usage

### Evaluating with GPT-4

```bash
python test-gpt4.py
```

### Evaluating with LLaMA

```bash
python test-llama.py
```

Both scripts will process the `refined_cleaned_data.jsonl` dataset and output the detection results, indicating the model's assessment of each article's authenticity.

## Dataset Details

- **Format**: JSON Lines (`.jsonl`)
- **Structure**:
  - `text`: The content of the news article.
  - `label`: Indicates authenticity (`"real"` or `"fake"`).

Ensure the dataset is placed in the root directory of the project for seamless access by the evaluation scripts.

## Report

The `Fake_News_Detection_With_Advanced_LLMs.pdf` document provides an in-depth analysis of the project's objectives, methodologies, experiments, and conclusions. It offers valuable insights into the performance of GPT-4 and LLaMA in fake news detection tasks.

## Contributing

Contributions are welcome! If you have suggestions, improvements, or encounter issues, please open an [issue](https://github.com/ty-berg/LLM-Fake-News-Detection/issues) or submit a [pull request](https://github.com/ty-berg/LLM-Fake-News-Detection/pulls).

## License

This project is licensed under the [MIT License](LICENSE).
