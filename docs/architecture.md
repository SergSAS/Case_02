# Architecture Documentation

## System Overview

The Text Classification Analysis system follows a modular architecture with clear separation of concerns. Each module is responsible for a specific aspect of the analysis pipeline.

## Component Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   main.py       │────▶│    Config       │────▶│  Environment    │
│ (Entry Point)   │     │   Manager       │     │   Variables     │
└────────┬────────┘     └─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   DataLoader    │────▶│  EurLex-57K     │
│                 │     │   Dataset       │
└────────┬────────┘     └─────────────────┘
         │
         ├──────────────────┬──────────────────┐
         ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Traditional    │ │      LLM        │ │   Evaluator     │
│   Classifier    │ │   Classifier    │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Visualizer     │
                   │                 │
                   └─────────────────┘
```

## Module Descriptions

### 1. Configuration Module (`src/config.py`)

**Purpose:** Centralized configuration management

**Responsibilities:**
- Load environment variables from `.env` file
- Provide API key access
- Define model hyperparameters
- Set up project directories
- Configure logging

**Key Classes:**
- `Config`: Main configuration class

**Dependencies:**
- `python-dotenv`
- Standard library modules

### 2. Data Loader Module (`src/data_loader.py`)

**Purpose:** Dataset management and preprocessing

**Responsibilities:**
- Download EurLex-57K dataset from HuggingFace
- Cache dataset locally for efficiency
- Create binary classification from multi-class data
- Calculate dataset statistics
- Provide train/test splits

**Key Classes:**
- `DataLoader`: Dataset loading and preprocessing

**Dependencies:**
- `datasets` (HuggingFace)
- `numpy`
- `pickle` (for caching)

### 3. Traditional ML Module (`src/traditional_ml.py`)

**Purpose:** Traditional machine learning classification

**Responsibilities:**
- TF-IDF text vectorization
- Logistic Regression training
- Model evaluation
- Feature importance analysis
- Model persistence

**Key Classes:**
- `TraditionalClassifier`: TF-IDF + Logistic Regression implementation

**Dependencies:**
- `scikit-learn`
- `joblib`
- `numpy`

### 4. LLM Classifier Module (`src/llm_classifier.py`)

**Purpose:** Large Language Model based classification

**Responsibilities:**
- Zero-shot classification
- Few-shot classification
- API rate limiting
- Cost estimation
- Response parsing

**Key Classes:**
- `LLMClassifier`: LLM-based classification wrapper

**Dependencies:**
- `groq` (API client)
- `openai` (optional)
- `scikit-learn` (metrics)

### 5. Evaluation Module (`src/evaluation.py`)

**Purpose:** Model comparison and analysis

**Responsibilities:**
- Compare multiple classification methods
- Generate comprehensive metrics
- Create comparison tables
- Provide recommendations
- Export results

**Key Classes:**
- `Evaluator`: Method comparison and analysis

**Dependencies:**
- `pandas`
- `rich` (terminal output)
- Standard library modules

### 6. Visualization Module (`src/visualization.py`)

**Purpose:** Results visualization

**Responsibilities:**
- Plot confusion matrices
- Create performance comparisons
- Generate speed/cost analysis
- Visualize feature importance
- Create summary reports

**Key Classes:**
- `Visualizer`: Chart and plot generation

**Dependencies:**
- `matplotlib`
- `seaborn`
- `numpy`

## Data Flow

1. **Initialization Phase:**
   - Load configuration from environment
   - Set up logging
   - Create output directories

2. **Data Loading Phase:**
   - Check for cached dataset
   - Download if necessary
   - Create binary classification
   - Generate statistics

3. **Training Phase (Traditional ML):**
   - Vectorize training texts
   - Train Logistic Regression
   - Save trained models

4. **Classification Phase:**
   - Traditional: Transform texts and predict
   - LLM: Send API requests with rate limiting

5. **Evaluation Phase:**
   - Calculate metrics for each method
   - Compare results
   - Generate insights

6. **Output Phase:**
   - Save JSON results
   - Create visualizations
   - Display summary tables

## Design Patterns

### 1. **Strategy Pattern**
Different classification strategies (Traditional vs LLM) implement similar interfaces for easy swapping.

### 2. **Factory Pattern**
Configuration creates appropriate classifiers based on settings.

### 3. **Singleton Pattern**
Configuration object is initialized once and shared across modules.

### 4. **Observer Pattern**
Logging system observes and records events across all modules.

## Error Handling

1. **API Failures:**
   - Graceful degradation with mock results
   - Exponential backoff for rate limits
   - Clear error messages

2. **Data Issues:**
   - Validation of dataset structure
   - Fallback to cached data
   - Sample size limits

3. **Configuration Errors:**
   - Default values for missing settings
   - Validation of API keys
   - Clear setup instructions

## Performance Considerations

1. **Caching:**
   - Dataset cached locally after first download
   - Model artifacts saved for reuse
   - Results stored in JSON for quick access

2. **Batch Processing:**
   - LLM requests processed in configurable batches
   - Vectorization done on full datasets
   - Parallel processing where applicable

3. **Memory Management:**
   - Sparse matrix representation for TF-IDF
   - Streaming dataset loading
   - Garbage collection after large operations

## Security Considerations

1. **API Key Management:**
   - Keys stored in `.env` file
   - Never committed to version control
   - Environment variable access only

2. **Data Privacy:**
   - No sensitive data in logs
   - Configurable output locations
   - Clean separation of code and data

## Extensibility

The architecture supports easy extension through:

1. **New Classifiers:**
   - Implement classifier interface
   - Add to main.py options
   - Automatic metric calculation

2. **New Datasets:**
   - Modify DataLoader for new sources
   - Maintain same output format
   - Reuse all downstream components

3. **New Visualizations:**
   - Add methods to Visualizer
   - Automatic integration with reports
   - Consistent styling

## Testing Strategy

1. **Unit Tests:**
   - Each module tested independently
   - Mock external dependencies
   - Fixture data for consistency

2. **Integration Tests:**
   - Full pipeline execution
   - API mock for LLM tests
   - Performance benchmarks

3. **End-to-End Tests:**
   - Complete analysis runs
   - Result validation
   - Visualization generation 