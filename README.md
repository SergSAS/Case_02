# Text Classification Analysis: Traditional ML vs LLM

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive framework for comparing traditional machine learning and Large Language Model (LLM) approaches to text classification using the EurLex-57K legal documents dataset.

## 📊 Project Overview

This project provides a detailed analysis and comparison of:
- **Traditional ML**: TF-IDF vectorization with Logistic Regression
- **LLM Approaches**: Zero-shot and few-shot classification using Groq API

### Key Features

- 🚀 Modular architecture with clean separation of concerns
- 📈 Comprehensive performance metrics (accuracy, F1-score, speed, cost)
- 🎨 Beautiful visualizations for results comparison
- 🔧 Easy configuration via environment variables
- 📝 Detailed logging and error handling
- 🌐 Support for multiple LLM providers (Groq, OpenAI)

## 🏗️ Architecture

```
text-classification-analysis/
├── src/                    # Core modules
│   ├── config.py          # Configuration management
│   ├── data_loader.py     # Dataset loading and preprocessing
│   ├── traditional_ml.py  # Traditional ML classifier
│   ├── llm_classifier.py  # LLM-based classifier
│   ├── evaluation.py      # Model evaluation and comparison
│   └── visualization.py   # Results visualization
├── config/                # Configuration files
│   ├── .env              # Environment variables (create from .env.example)
│   └── .env.example      # Example configuration
├── data/                  # Dataset cache
├── results/               # Analysis results
├── logs/                  # Application logs
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
└── FINAL_REPORT.md      # Detailed analysis report
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Groq API key for LLM classification

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/text-classification-analysis.git
   cd text-classification-analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp config/.env.example config/.env
   # Edit config/.env and add your API keys
   ```

### Basic Usage

Run the full analysis:
```bash
python main.py
```

Run specific analysis modes:
```bash
# Traditional ML only
python main.py --mode traditional

# LLM analysis only
python main.py --mode llm --llm-samples 50

# Compare existing results
python main.py --mode compare

# Generate visualizations
python main.py --visualize
```

## 📖 Detailed Documentation

### Configuration Options

All configuration is managed through environment variables in `config/.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for LLM access | Required for LLM |
| `LLM_MODEL` | LLM model to use | `gemma2-9b-it` |
| `LLM_TEMPERATURE` | Generation temperature | `0.1` |
| `RATE_LIMIT_DELAY` | Delay between API calls | `2.0` seconds |

### Command Line Arguments

```bash
python main.py [OPTIONS]

Options:
  --mode {full,traditional,llm,compare}
                        Analysis mode to run (default: full)
  --config CONFIG       Path to configuration file (default: config/.env)
  --force-reload        Force reload dataset from source
  --llm-samples N       Number of samples for LLM classification (default: 20)
  --visualize           Generate visualization plots
  --output-dir DIR      Override output directory for results
```

### Module Documentation

#### Config Module (`src/config.py`)
Manages all configuration settings, API keys, and project paths.

```python
from src import Config

config = Config("config/.env")
print(config.groq_api_key)
print(config.model_config)
```

#### DataLoader Module (`src/data_loader.py`)
Handles EurLex-57K dataset loading and preprocessing.

```python
from src import DataLoader

loader = DataLoader(config)
data = loader.load_dataset()
stats = loader.get_statistics(data)
```

#### Traditional ML Module (`src/traditional_ml.py`)
Implements TF-IDF + Logistic Regression classification.

```python
from src import TraditionalClassifier

classifier = TraditionalClassifier(config)
classifier.train(train_texts, train_labels)
metrics = classifier.evaluate(test_texts, test_labels)
```

#### LLM Classifier Module (`src/llm_classifier.py`)
Provides zero-shot and few-shot classification using LLMs.

```python
from src import LLMClassifier

llm = LLMClassifier(config)
predictions = llm.classify_zero_shot(texts)
metrics = llm.evaluate(predictions, true_labels)
```

## 📊 Results Summary

Based on our analysis of the EurLex-57K dataset:

| Method | Accuracy | F1-Score | Speed | Cost |
|--------|----------|----------|-------|------|
| Traditional ML | 88.90% | 0.850 | 573K docs/s | $0 |
| LLM Zero-shot | 25.00% | 0.400 | 0.3 docs/s | $0.30/1K |
| LLM Few-shot | 20.00% | 0.333 | 0.2 docs/s | $0.50/1K |

**Key Findings:**
- Traditional ML excels in speed and cost-efficiency
- LLMs struggle with the specific binary classification task
- Hybrid approaches may offer the best balance

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

This project uses:
- [Black](https://github.com/psf/black) for code formatting
- [Flake8](https://flake8.pycqa.org/) for linting
- [mypy](http://mypy-lang.org/) for type checking

Run all checks:
```bash
black src/
flake8 src/
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [EurLex-57K Dataset](https://huggingface.co/datasets/pietrolesci/eurlex-57k) by Pietro Lesci
- [Groq](https://groq.com/) for LLM API access
- [scikit-learn](https://scikit-learn.org/) for ML algorithms
- [Hugging Face](https://huggingface.co/) for dataset hosting

## 📮 Contact

For questions or feedback, please open an issue on GitHub or contact [your-email@example.com](mailto:your-email@example.com).

---

Made with ❤️ for the ML community 