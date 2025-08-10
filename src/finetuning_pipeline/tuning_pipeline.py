from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from logger.logger import get_logger
import traceback

logger = get_logger("FineTuningPipeline")


def lora_configuration(base_model, lora_config):
    """Configure LoRA parameters and apply to base model"""
    try:
        logger.info("Configuring LoRA parameters")
        logger.info(f"LoRA configuration:")
        logger.info(f"  - R: {lora_config['R']}")
        logger.info(f"  - Alpha: {lora_config['LORA_ALPHA']}")
        logger.info(f"  - Dropout: {lora_config['LORA_DROPOUT']}")
        logger.info(f"  - Bias: {lora_config['BIAS']}")
        logger.info(f"  - Task type: {lora_config['TASK_TYPE']}")
        logger.info(f"  - Use RSLoRA: {lora_config['USE_RSLORA']}")
        logger.info(f"  - Init weights: {lora_config['INIT_LORA_WEIGHTS']}")
        logger.info(f"  - Target modules: {lora_config['TARGET_MODULES']}")

        # Apply LoRA using PEFT
        lora_config_dict = LoraConfig(
            r=lora_config["R"],
            lora_alpha=lora_config["LORA_ALPHA"],
            lora_dropout=lora_config["LORA_DROPOUT"],
            bias=lora_config["BIAS"],
            task_type=lora_config["TASK_TYPE"],
            use_rslora=lora_config["USE_RSLORA"],
            init_lora_weights=lora_config["INIT_LORA_WEIGHTS"],
            target_modules=lora_config["TARGET_MODULES"]
        )
        
        logger.info("Applying LoRA configuration to base model...")
        model = get_peft_model(base_model, lora_config_dict)
        logger.info("LoRA configuration applied successfully")
        
        return model

    except Exception as e:
        logger.error(f"Failed to configure LoRA: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def training_lora(model, tokenized_text, tokenizer, lora_config):
    """Train the LoRA model with comprehensive logging and error handling"""
    try:
        logger.info("Starting LoRA training process")
        logger.info(f"Training configuration:")
        logger.info(f"  - Output directory: {lora_config['OUTPUT_DIR']}")
        logger.info(f"  - Number of epochs: {lora_config['NUM_TRAIN_EPOCHS']}")
        logger.info(f"  - Batch size: {lora_config['PER_DEVICE_TRAIN_BATCH_SIZE']}")
        logger.info(f"  - Gradient accumulation steps: {lora_config['GRADIENT_ACCUMULATION_STEPS']}")
        logger.info(f"  - Learning rate: {lora_config['LEARNING_RATE']}")
        logger.info(f"  - FP16: {lora_config['FP16']}")
        logger.info(f"  - Logging steps: {lora_config['LOGGING_STEPS']}")
        logger.info(f"  - Save steps: {lora_config['SAVE_STEPS']}")
        
        # Step 6: TrainingArguments
        logger.info("Creating training arguments...")
        training_args = TrainingArguments(
            output_dir=lora_config["OUTPUT_DIR"],
            num_train_epochs=lora_config["NUM_TRAIN_EPOCHS"],
            per_device_train_batch_size=lora_config["PER_DEVICE_TRAIN_BATCH_SIZE"],
            gradient_accumulation_steps=lora_config["GRADIENT_ACCUMULATION_STEPS"],
            learning_rate=lora_config["LEARNING_RATE"],
            fp16=lora_config["FP16"],
            logging_steps=lora_config["LOGGING_STEPS"],
            save_steps=lora_config["SAVE_STEPS"],
            save_total_limit=lora_config["SAVE_TOTAL_LIMIT"],
            logging_dir=lora_config["LOGGING_DIR"],
            report_to=lora_config["REPORT_TO"],
            label_names=lora_config["LABEL_NAMES"]
        )
        logger.info("Training arguments created successfully")

        # Step 7: Trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_text["train"],
            eval_dataset=tokenized_text["test"]
        )
        logger.info("Trainer initialized successfully")

        # Step 8: Train
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully")

        # Save model and tokenizer
        logger.info("Saving model and tokenizer...")
        model.save_pretrained(lora_config["MODEL_SAVE_DIR"])
        tokenizer.save_pretrained(lora_config["TOKENIZER_SAVE_DIR"])
        logger.info(f"Model saved to: {lora_config['MODEL_SAVE_DIR']}")
        logger.info(f"Tokenizer saved to: {lora_config['TOKENIZER_SAVE_DIR']}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def fine_tuning_pipeline(tokenized_text, tokenizer, base_model, lora_config):
    """Complete fine-tuning pipeline with error handling and logging"""
    try:
        logger.info("Starting complete fine-tuning pipeline")
        
        # Configure LoRA
        model = lora_configuration(base_model, lora_config)
        
        # Train the model
        training_lora(model, tokenized_text, tokenizer, lora_config)
        
        logger.info("Fine-tuning pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Fine-tuning pipeline failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
