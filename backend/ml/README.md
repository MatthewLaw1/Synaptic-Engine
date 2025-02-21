# Enhanced ML Training Pipeline for Thought Classification

This implementation expands the thought classification capabilities using a multi-modal approach combining EEG and biometric data. The system uses sentiment analysis as an initial filter and employs an iterative ML model stack to progressively refine thought predictions.

## Architecture Overview

### 1. Data Processing
- **EEG Processing** (`eeg_processing.py`)
  - Enhanced feature extraction including:
    - FFT and PSD features
    - Wavelet decomposition
    - Entropy measures
    - Hjorth parameters
    - Channel connectivity metrics

- **Biometric Processing** (`biometric_processing.py`)
  - Heart Rate Variability (HRV) analysis
  - Blood pressure processing
  - Galvanic skin response features
  - Respiratory signal analysis

### 2. Sentiment Analysis (`sentiment_analysis.py`)
- Initial filtering layer using emotional context
- Multi-head attention for temporal features
- Combined EEG and biometric feature processing
- Outputs valence and arousal scores

### 3. Model Architecture (`models.py`)
- **Enhanced EEG Embedding**
  - Deeper CNN architecture
  - Self-attention mechanisms
  - Increased embedding dimension (256)

- **Multi-Modal Fusion**
  - Adaptive fusion of EEG and biometric embeddings
  - Gated integration mechanism

- **Iterative Classification**
  - Progressive refinement of predictions
  - Confidence-based filtering
  - Enhanced triplet loss with adaptive margin

### 4. Training System (`train.py`)
- Curriculum learning implementation
- Multi-task training support
- Hyperparameter optimization
- Comprehensive validation metrics

### 5. Inference Pipeline (`inference.py`)
- Complete prediction pipeline
- Sentiment-based filtering
- Confidence scoring
- Prediction explanation system

## Usage

### Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### Training

```python
from ml.train import ModelTrainer, prepare_data
from ml.models import IterativeModelStack
from ml.sentiment_analysis import create_sentiment_analyzer

# Prepare data
(train_data, val_data, test_data) = prepare_data(
    eeg_features,
    bio_features,
    thought_labels,
    sentiment_labels
)

# Initialize models
model_stack = IterativeModelStack(
    eeg_feature_dim=eeg_feature_dim,
    bio_feature_dim=bio_feature_dim
)

sentiment_analyzer = create_sentiment_analyzer(
    eeg_feature_dim=eeg_feature_dim,
    bio_feature_dim=bio_feature_dim
)

# Create trainer
trainer = ModelTrainer(
    model_stack=model_stack,
    sentiment_analyzer=sentiment_analyzer
)

# Train models
sentiment_history = trainer.train_sentiment_analyzer(
    train_loader,
    val_loader
)

stack_history = trainer.train_model_stack(
    train_loader,
    val_loader
)
```

### Inference

```python
from ml.inference import load_pipeline

# Load pipeline
pipeline = load_pipeline(
    model_path='path/to/model.pt',
    sentiment_model_path='path/to/sentiment.pt',
    thought_labels_path='path/to/labels.npy'
)

# Make predictions
predictions = pipeline.predict(
    eeg_data=eeg_signal,
    rr_intervals=hrv_data,
    blood_pressure=(systolic, diastolic),
    gsr_signal=gsr_data,
    resp_signal=resp_data
)

# Get explanations
for pred in predictions:
    explanation = pipeline.explain_prediction(pred)
    print(explanation)
```

## Performance Considerations

- The system can theoretically handle hundreds to thousands of distinct thoughts
- Practical limitations include:
  - Signal-to-noise ratio in input data
  - Similarity between thought patterns
  - Available training data
  - Computational resources

## Future Improvements

1. Online Learning
   - Implement continuous model updates
   - Add user feedback integration
   - Develop adaptive thresholds

2. Model Optimization
   - Explore quantization for faster inference
   - Implement model pruning
   - Add distributed training support

3. Feature Engineering
   - Add more advanced connectivity metrics
   - Implement real-time signal quality checks
   - Explore additional biometric signals

4. Validation
   - Add cross-subject validation
   - Implement model interpretability tools
   - Add performance monitoring system