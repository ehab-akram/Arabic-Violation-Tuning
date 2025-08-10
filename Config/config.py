from pathlib import Path
from peft import TaskType
import logging

BASE_DIR = Path(__file__).resolve().parent.parent


ModelConfig = {
    "MODEL_NAME": "UBC-NLP/MARBERTv2",
    "CACHE_DIR": None,  # or "./cache"
    "USE_FAST": True,
    "NUM_LABELS": 2,  # Binary classification
    "TRUST_REMOTE_CODE": True,
    "LOCAL_FILES_ONLY": False,
    "PADDING_SIDE": "right",
    "TRUNCATION_SIDE": "right",
    "MAX_LEN": 512,
    "OUTPUT_ATTENTIONS": False,
    "OUTPUT_HIDDEN_STATES": False,
    "TORCH_DTYPE": "auto",
    "LOW_CPU_MEM_USAGE": True,
    "DEVICE_MAP": "auto",
    "IGNORE_MISMATCHED_SIZES": True
}


DataConfig = {
    # load_data
    "FILEPATH_OR_BUFFER": BASE_DIR / "Dataset" / "raw" / "offensive_Dataset.csv",
    "ENCODING": "utf-8",
    "SEP": ",",  # CSV standard
    "NAMES": None,  # Since you already have headers
    "DTYPE": None,  # Let pandas infer unless you have specific needs
    "ENGINE": "python",  # Safer for multilingual/complex text
    "ENCODING_ERRORS": "strict",  # Default; use "replace" or "ignore" if issues arise
    "SKIPROWS": 0, # Don't skip any rows



    "TEST_SIZE":0.1,
    "TRAIN_SIZE":0.9,
    "KEEP_IN_MEMORY":True,
    "SEED":42,

    "BATCHED":True,



    "TRUNCATION" : True,
    "PADDING":"max_length",
    "MAX_LENGTH":512,

    "TYPE":"torch",
    "COLUMNS":["input_ids", "attention_mask", "labels"],


}


LoRAConfig = {
    # LoRA Parameters
    "R": 8,  # Commonly used low-rank size
    "LORA_ALPHA": 16,  # Scaling factor; often 2x R
    "LORA_DROPOUT": 0.01,  # Regular dropout rate
    "BIAS": "none",  # Typically "none", unless you want to fine-tune bias terms too
    "TASK_TYPE": TaskType.SEQ_CLS,  # Sequence classification task
    "USE_RSLORA": False,  # Leave as False unless you specifically want Robust & Scalable LoRA
    "INIT_LORA_WEIGHTS": "gaussian",  # Use 'gaussian' or 'bert' for better initialization
    "TARGET_MODULES": ["query", "value"],  # Common for BERT-like models (like MARBERTv2)

    # TrainingArguments
    "OUTPUT_DIR": "./marbertv2_lora_finetuned",
    "NUM_TRAIN_EPOCHS": 3,  # Increase if needed
    "PER_DEVICE_TRAIN_BATCH_SIZE": 4,  # Adjust based on GPU memory
    "GRADIENT_ACCUMULATION_STEPS": 2,  # Effective batch size = 4 x 2 = 8
    "LEARNING_RATE": 2e-4,  # Good for LoRA tuning
    "FP16": True,  # Use FP16 if your GPU supports it (saves memory)
    "LOGGING_STEPS": 10,
    "SAVE_STEPS": 100,
    "SAVE_TOTAL_LIMIT": 2,
    "LOGGING_DIR": "./logs",
    "REPORT_TO": "none",  # or "tensorboard" or "wandb" or "none" if you want logging
    "LABEL_NAMES": ["labels"],

    # Save paths
    "TOKENIZER_SAVE_DIR": "marbertv2-lora-adapter",
    "MODEL_SAVE_DIR": "marbertv2-lora-adapter",
}


LoggerConfig = {
    # Log directory settings
    "LOG_DIR": "../../logger/logs",
    "LOG_RETENTION_DAYS": 6,
    
    # Logging levels
    "DEFAULT_LOG_LEVEL": logging.DEBUG,
    "CONSOLE_LOG_LEVEL": logging.INFO,
    
    # Log format
    "LOG_FORMAT": '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    
    # File handling
    "LOG_FILE_MODE": 'a',
    "LOG_FILE_ENCODING": 'utf-8',
    
    # Date format for log files
    "DATE_FORMAT": "%Y-%m-%d",
    "LOG_FILE_EXTENSION": ".log"
}


