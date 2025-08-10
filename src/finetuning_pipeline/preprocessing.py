from datasets import Dataset
import pandas as pd
from logger.logger import get_logger
import traceback

logger = get_logger("DataPreprocessing")


def load_data(DataConfig):
    """Load dataset from CSV with error handling and logging"""
    try:
        logger.info("Loading dataset from CSV file")
        logger.info(f"File path: {DataConfig['FILEPATH_OR_BUFFER']}")
        
        df = pd.read_csv(
            filepath_or_buffer=DataConfig["FILEPATH_OR_BUFFER"],
            encoding=DataConfig["ENCODING"],
            sep=DataConfig["SEP"],
            names=DataConfig["NAMES"],
            dtype=DataConfig["DTYPE"],
            engine=DataConfig["ENGINE"],
            encoding_errors=DataConfig["ENCODING_ERRORS"],
            skiprows=DataConfig["SKIPROWS"]
        )
        
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Data types: {df.dtypes.to_dict()}")
        
        return df
        
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {DataConfig['FILEPATH_OR_BUFFER']}")
        raise
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def HuggingFace_dataset(df, DataConfig):
    """Convert pandas DataFrame to HuggingFace Dataset with train/test split"""
    try:
        logger.info("Converting pandas DataFrame to HuggingFace Dataset")
        logger.info(f"Original DataFrame shape: {df.shape}")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(df)
        logger.info(f"Dataset created successfully. Size: {len(dataset)}")
        
        # Perform train/test split
        logger.info("Performing train/test split")
        logger.info(f"Test size: {DataConfig['TEST_SIZE']}")
        logger.info(f"Train size: {DataConfig['TRAIN_SIZE']}")
        logger.info(f"Seed: {DataConfig['SEED']}")
        
        dataset = dataset.train_test_split(
            test_size=DataConfig["TEST_SIZE"],
            seed=DataConfig["SEED"],
            train_size=DataConfig["TRAIN_SIZE"],
            keep_in_memory=DataConfig["KEEP_IN_MEMORY"],
        )
        
        logger.info(f"Train dataset size: {len(dataset['train'])}")
        logger.info(f"Test dataset size: {len(dataset['test'])}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to create HuggingFace dataset: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise



def data_tokenization(tokenizer, dataset, DataConfig):
    """Tokenize dataset with error handling and logging"""
    try:
        logger.info("Starting data tokenization")
        logger.info(f"Tokenization parameters:")
        logger.info(f"  - Truncation: {DataConfig['TRUNCATION']}")
        logger.info(f"  - Padding: {DataConfig['PADDING']}")
        logger.info(f"  - Max length: {DataConfig['MAX_LENGTH']}")
        logger.info(f"  - Batched: {DataConfig['BATCHED']}")
        
        def tokenize(batch):
            """Tokenize a batch of texts"""
            try:
                # Drop any non-str values, or coerce to str()
                texts = [
                    t if isinstance(t, str) else ""
                    for t in batch["text"]
                ]
                return tokenizer(
                    texts,
                    truncation=DataConfig["TRUNCATION"],
                    padding=DataConfig["PADDING"],
                    max_length=DataConfig["MAX_LENGTH"]
                )
            except Exception as e:
                logger.error(f"Error in tokenize function: {str(e)}")
                raise

        logger.info("Applying tokenization to dataset...")
        tokenized = dataset.map(
            tokenize,
            batched=DataConfig["BATCHED"]
        )
        logger.info("Tokenization completed successfully")

        logger.info("Setting dataset format...")
        tokenized.set_format(
            type=DataConfig["TYPE"],
            columns=DataConfig["COLUMNS"]
        )
        logger.info(f"Dataset format set to: {DataConfig['TYPE']}")
        logger.info(f"Columns: {DataConfig['COLUMNS']}")

        return tokenized
        
    except Exception as e:
        logger.error(f"Failed to tokenize dataset: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def preprocessing_pipeline(tokenizer, DataConfig):
    """Complete preprocessing pipeline with error handling and logging"""
    try:
        logger.info("Starting complete preprocessing pipeline")
        
        # Load data
        df = load_data(DataConfig)
        
        # Convert to HuggingFace dataset
        dataset = HuggingFace_dataset(df, DataConfig)
        
        # Tokenize data
        tokenized = data_tokenization(tokenizer, dataset, DataConfig)
        
        logger.info("Preprocessing pipeline completed successfully")
        return tokenized
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


