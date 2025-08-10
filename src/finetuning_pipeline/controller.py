from Config.config import ModelConfig, DataConfig, LoRAConfig
from src.finetuning_pipeline.tuning_pipeline import fine_tuning_pipeline
from src.finetuning_pipeline.load_model import setup_model
from src.finetuning_pipeline.preprocessing import preprocessing_pipeline
from logger.logger import get_logger
import traceback


class FineTuningController:
    """Main controller for the fine-tuning process"""

    def __init__(self):
        # Initialize logger
        self.logger = get_logger("FineTuningController")
        
        # config passing to the function
        self.model_config = ModelConfig.copy()
        self.data_config = DataConfig.copy()
        self.lora_config = LoRAConfig.copy()
        
        self.logger.info("FineTuningController initialized with configurations")

    def run_complete_pipeline(self):
        """Run the complete fine-tuning pipeline with error handling and logging"""
        try:
            self.logger.info("Starting complete fine-tuning pipeline")
            
            # Load the Model and Tokenizer
            self.logger.info("Loading model and tokenizer...")
            tokenizer, base_model = setup_model(self.model_config)
            self.logger.info("Model and tokenizer loaded successfully")
            
            # Preprocess data
            self.logger.info("Starting data preprocessing...")
            tokenized_text = preprocessing_pipeline(tokenizer, self.data_config)
            self.logger.info("Data preprocessing completed successfully")
            
            # Fine-tune the model
            self.logger.info("Starting fine-tuning process...")
            fine_tuning_pipeline(tokenized_text, tokenizer, base_model, self.lora_config)
            self.logger.info("Fine-tuning pipeline completed successfully")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise



def main():
    try:
        controller = FineTuningController()
        controller.run_complete_pipeline()
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        return 1
    return 0


if __name__ == "__main__":
    main()
