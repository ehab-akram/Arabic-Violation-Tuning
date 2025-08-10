from transformers import AutoTokenizer, AutoModelForSequenceClassification
from logger.logger import get_logger
import traceback

logger = get_logger("ModelManager")


def setup_model(model_config):
    """Setup model and tokenizer with comprehensive error handling and logging"""
    try:
        logger.info("Starting model and tokenizer setup")
        logger.info(f"Model name: {model_config['MODEL_NAME']}")
        logger.info(f"Number of labels: {model_config['NUM_LABELS']}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["MODEL_NAME"],
            cache_dir=model_config["CACHE_DIR"],
            use_fast=model_config["USE_FAST"],
            trust_remote_code=model_config["TRUST_REMOTE_CODE"],
            local_files_only=model_config["LOCAL_FILES_ONLY"],
            padding_side=model_config["PADDING_SIDE"],
            truncation_side=model_config["TRUNCATION_SIDE"],
            max_len=model_config["MAX_LEN"],
        )
        logger.info("Tokenizer loaded successfully")
        
        # Load base model
        logger.info("Loading base model...")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_config["MODEL_NAME"],
            num_labels=model_config["NUM_LABELS"],
            cache_dir=model_config["CACHE_DIR"],
            trust_remote_code=model_config["TRUST_REMOTE_CODE"],
            local_files_only=model_config["LOCAL_FILES_ONLY"],
            output_attentions=model_config["OUTPUT_ATTENTIONS"],
            output_hidden_states=model_config["OUTPUT_HIDDEN_STATES"],
            torch_dtype=model_config["TORCH_DTYPE"],
            low_cpu_mem_usage=model_config["LOW_CPU_MEM_USAGE"],
            device_map=model_config["DEVICE_MAP"],
            ignore_mismatched_sizes=model_config["IGNORE_MISMATCHED_SIZES"]
        )
        logger.info("Base model loaded successfully")
        
        logger.info("Model and tokenizer setup completed successfully")
        return tokenizer, base_model
        
    except Exception as e:
        logger.error(f"Failed to setup model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


