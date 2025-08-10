# Violation Detection Fine-tuning Pipeline

A comprehensive fine-tuning pipeline for Arabic violation detection using MARBERTv2 with Unsloth optimizations and LoRA. This project provides both training and inference capabilities for Arabic text classification.

## 📹 Demo Video

Watch the Arabic Violation Detection system in action:

[https://github.com/your-username/violation_tuning/assets/your-user-id/Arabic_violation_Check.mp4](https://github.com/ehab-akram/Arabic-Violation-Tuning/blob/main/Arabic_violation_Check.mp4)

*Note: If the video doesn't play, you can download it from the repository root directory.*

## 🚀 Features

- **Complete Pipeline**: End-to-end fine-tuning from data preprocessing to model evaluation
- **Unsloth Integration**: Optimized training with 4-bit quantization and memory efficiency  
- **LoRA Support**: Parameter-efficient fine-tuning with Low-Rank Adaptation
- **Inference Pipeline**: Ready-to-use inference application with web interface
- **Pre-trained Models**: Includes fine-tuned MARBERTv2 checkpoints
- **Comprehensive Evaluation**: Detailed metrics, error analysis, and visualization
- **Modular Design**: Clean, maintainable code with clear separation of concerns
- **Robust Logging**: Comprehensive logging with component-specific log files
- **Memory Management**: Automatic memory monitoring and optimization

## 📁 Project Structure

```
violation_tuning/
├── Config/
│   └── config.py                 # Configuration classes
├── Dataset/
│   └── raw/
│       ├── offensive_Dataset.csv # Raw training data
│       └── offensive_Dataset.csv~# Backup file
├── logger/
│   ├── __init__.py
│   ├── logger.py                 # Logging utilities
│   └── logs/                     # Component-specific log files
│       ├── all/                  # Master logs
│       ├── DataPreprocessing/    # Preprocessing logs
│       ├── FineTuningController/ # Controller logs
│       ├── FineTuningPipeline/   # Pipeline logs
│       └── ModelManager/         # Model management logs
├── src/
│   ├── __init__.py
│   ├── finetuning_pipeline/      # Training pipeline
│   │   ├── __init__.py
│   │   ├── controller.py         # Main pipeline orchestrator
│   │   ├── load_model.py         # Model loading utilities
│   │   ├── preprocessing.py      # Data preprocessing
│   │   ├── tuning_pipeline.py    # Core training pipeline
│   │   ├── marbertv2_lora_finetuned/  # Training checkpoints
│   │   │   ├── checkpoint-4500/
│   │   │   └── checkpoint-4596/
│   │   └── marbertv2-lora-adapter/    # Final LoRA adapter
│   └── infrence_pipline/         # Inference pipeline
│       ├── __init__.py
│       ├── inference_app.py      # Web-based inference application
│       ├── model_feedback.xlsx   # Model evaluation results
│       ├── prediction_results.xlsx # Prediction outputs
│       └── templates/
│           └── index.html        # Web interface
├── requirements.txt              # Dependencies
└── README.md                    # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd violation_tuning
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   # Test the training pipeline
   python src/finetuning_pipeline/controller.py --help
   
   # Or test the inference application
   python src/infrence_pipline/inference_app.py --help
   ```

## 📊 Data Format

The pipeline expects Arabic text data in the following format:

- **Input**: CSV file (`offensive_Dataset.csv`) with columns:
  - `Text`: Arabic text content
  - `Label`: Binary labels (0: non-offensive, 1: offensive)

- **Output**: Processed and split datasets for training, validation, and testing

## ⚙️ Configuration

All configuration is centralized in `Config/config.py`:

### Data Configuration (`DataConfig`)
- File paths and data splitting ratios
- Text preprocessing parameters
- Arabic text normalization settings

### Model Configuration (`ModelConfig`)
- MARBERTv2 model settings
- Tokenization parameters
- Device and memory settings

### Training Configuration (`TrainingConfig`)
- Training hyperparameters
- Evaluation and saving strategies
- Optimization settings

### LoRA Configuration (`LoRAConfig`)
- LoRA parameters (rank, alpha, dropout)
- Unsloth optimization settings
- Target modules for adaptation

## 🚀 Usage

### Quick Start

#### Training Pipeline
Run the complete fine-tuning pipeline:

```bash
python src/finetuning_pipeline/controller.py
```

#### Inference Application
Run the web-based inference application:

```bash
python src/infrence_pipline/inference_app.py
```

Then open your browser and navigate to `http://localhost:5000` to use the web interface for text classification.

### Training Pipeline Usage

```python
from src.finetuning_pipeline.controller import ViolationDetectionPipeline

# Initialize pipeline
pipeline = ViolationDetectionPipeline()

# Run complete pipeline
results = pipeline.run_complete_pipeline(
    output_dir="violation_detection_output",
    force_reprocess=False
)

# Check results
if results['pipeline_success']:
    print(f"✅ Pipeline completed in {results['pipeline_time_formatted']}")
    print(f"📊 Final accuracy: {results['pipeline_results']['evaluation']['metrics']['basic_metrics']['accuracy']:.4f}")
```

### Inference Usage

```python
from src.infrence_pipline.inference_app import load_model, predict_text

# Load the trained model
model, tokenizer = load_model()

# Make predictions
text = "Your Arabic text here"
prediction = predict_text(text, model, tokenizer)
print(f"Prediction: {prediction}")
```

### Individual Components

#### Data Preprocessing
```python
from src.finetuning_pipeline.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
results = preprocessor.run_preprocessing_pipeline()
```

#### Model Loading
```python
from src.finetuning_pipeline.load_model import load_model_for_training

# Load model with Unsloth optimizations
model, tokenizer = load_model_for_training()
```

#### Training Pipeline
```python
from src.finetuning_pipeline.tuning_pipeline import FineTuningPipeline

# Initialize and run training
pipeline = FineTuningPipeline()
training_results = pipeline.run_training()
```

#### Using Pre-trained Model
```python
from src.finetuning_pipeline.load_model import load_trained_model

# Load the fine-tuned model
model, tokenizer = load_trained_model("marbertv2-lora-adapter")
```

## 🎯 Pre-trained Models

The project includes pre-trained MARBERTv2 models with LoRA adapters:

- **marbertv2-lora-adapter/**: Final fine-tuned LoRA adapter ready for inference
- **marbertv2_lora_finetuned/**: Training checkpoints including:
  - `checkpoint-4500/`: Intermediate training checkpoint
  - `checkpoint-4596/`: Latest training checkpoint

These models are specifically fine-tuned for Arabic violation detection and can be used directly for inference without additional training.

## 📈 Pipeline Stages

### 1. Data Preprocessing
- Load raw CSV data (`offensive_Dataset.csv`)
- Clean and normalize Arabic text
- Handle encoding and format issues
- Prepare data for training

### 2. Model Setup
- Load MARBERTv2 base model with Unsloth optimizations
- Apply LoRA configuration for parameter-efficient fine-tuning
- Configure 4-bit quantization for memory efficiency
- Setup tokenizer for Arabic text processing

### 3. Training
- Configure training parameters and optimization
- Run fine-tuning with LoRA adapters
- Monitor training progress and metrics
- Save checkpoints during training

### 4. Model Saving
- Save final LoRA adapter
- Export model configuration
- Prepare model for inference

### 5. Inference
- Load pre-trained model and adapter
- Process new Arabic text for classification
- Provide web interface for easy interaction
- Generate prediction results and feedback

## 📊 Evaluation Metrics

The pipeline provides comprehensive evaluation including:

- **Basic Metrics**: Accuracy, Precision, Recall, F1-Score
- **Per-Class Metrics**: Individual class performance
- **Advanced Metrics**: ROC AUC, Macro/Micro averages
- **Error Analysis**: False positives/negatives analysis
- **Visualizations**: Confusion matrix, ROC curves, metrics comparison

## 📁 Output Structure

After running the training pipeline, you'll find:

```
src/finetuning_pipeline/
├── marbertv2_lora_finetuned/    # Training checkpoints
│   ├── checkpoint-4500/         # Intermediate checkpoint
│   └── checkpoint-4596/         # Latest checkpoint
└── marbertv2-lora-adapter/      # Final LoRA adapter
    ├── adapter_config.json      # Adapter configuration
    ├── adapter_model.safetensors # LoRA weights
    ├── tokenizer files...       # Tokenizer configuration
    └── README.md               # Model information

logger/logs/                     # Training logs
├── all/                        # Master logs
├── DataPreprocessing/          # Preprocessing logs
├── FineTuningController/       # Controller logs
├── FineTuningPipeline/         # Pipeline logs
└── ModelManager/               # Model management logs
```

After running inference, you'll find:

```
src/infrence_pipline/
├── model_feedback.xlsx         # Model evaluation results
├── prediction_results.xlsx     # Prediction outputs
└── templates/
    └── index.html             # Web interface
```

## 🔧 Customization

### Modifying Configurations

Edit `Config/config.py` to customize:

- **Model**: Change `MODEL_NAME` for different base models
- **Training**: Adjust `NUM_EPOCHS`, `LEARNING_RATE`, batch sizes
- **LoRA**: Modify `LORA_R`, `LORA_ALPHA`, target modules
- **Data**: Change preprocessing parameters, split ratios

### Adding Custom Preprocessing

Extend the preprocessing in `preprocessing.py`:

```python
def custom_text_cleaning(self, text):
    # Your custom text preprocessing
    return cleaned_text
```

### Custom Model Loading

Modify `load_model.py` to support different models:

```python
def load_custom_model(model_name):
    # Custom model loading logic
    return model, tokenizer
```

### Extending Inference

Modify `inference_app.py` to add new features:

```python
def custom_prediction_logic(text, model, tokenizer):
    # Custom inference logic
    return enhanced_prediction
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `TrainingConfig`
   - Enable gradient checkpointing
   - Use smaller LoRA rank

2. **Import Errors**
   - Check Python path and dependencies
   - Verify all required packages are installed from `requirements.txt`

3. **Data Loading Issues**
   - Verify data file paths in `Config/config.py`
   - Check CSV file encoding (should be UTF-8 for Arabic text)
   - Ensure the CSV file has the correct column names (`Text`, `Label`)

4. **Inference Application Issues**
   - Check if the model adapter files exist in `marbertv2-lora-adapter/`
   - Verify Flask dependencies are installed
   - Ensure port 5000 is available

5. **Training Convergence**
   - Adjust learning rate
   - Increase warmup steps
   - Modify LoRA parameters

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 📝 Logging

The pipeline uses comprehensive logging:

- **Component-specific logs**: Each module has its own log file
- **Master log**: All logs are also written to a master log file
- **Automatic cleanup**: Logs older than 6 days are automatically removed
- **Structured format**: Timestamp, component, level, message

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2) for the base model
- [Unsloth](https://github.com/unslothai/unsloth) for optimization
- [HuggingFace](https://huggingface.co/) for the transformers library
- [LoRA](https://arxiv.org/abs/2106.09685) for parameter-efficient fine-tuning

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the logs for error details
3. Open an issue with detailed information
4. Include system information and error messages

---

**Happy Fine-tuning! 🚀**
