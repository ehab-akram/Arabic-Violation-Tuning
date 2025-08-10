import os
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import glob
sys.path.append(str(Path(__file__).parent.parent))

# Import configuration
from Config.config import LoggerConfig

# use --> logger = get_logger("Controller")

def ensure_log_directory_exists(LOG_DIR):
    """Ensure the logs directory exists."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)


def get_dated_log_handler(component_name, level=LoggerConfig["DEFAULT_LOG_LEVEL"]):
    """
    Creates a log handler that:
    1. Places logs in component-specific folders
    2. Uses date in filename (YYYY-MM-DD format)
    3. Cleans up logs older than configured days
    
    Args:
        component_name (str): Name of the component (used for folder name)
        level (int): Logging level
        
    Returns:
        logging.FileHandler: Configured file handler
    """
    # Create component directory if it doesn't exist
    LOG_DIR = LoggerConfig["LOG_DIR"]
    component_dir = os.path.join(LOG_DIR, component_name)
    if not os.path.exists(component_dir):
        os.makedirs(component_dir)
        
    # Get current date for filename
    current_date = datetime.now().strftime(LoggerConfig["DATE_FORMAT"])
    log_filename = f"{component_name}_{current_date}{LoggerConfig['LOG_FILE_EXTENSION']}"
    log_path = os.path.join(component_dir, log_filename)
    
    # Clean up old log files (older than configured days)
    cleanup_old_logs(component_dir, days=LoggerConfig["LOG_RETENTION_DAYS"])
    
    # Create and configure handler
    handler = logging.FileHandler(log_path, mode=LoggerConfig["LOG_FILE_MODE"], encoding=LoggerConfig["LOG_FILE_ENCODING"])
    formatter = logging.Formatter(LoggerConfig["LOG_FORMAT"])
    handler.setFormatter(formatter)
    handler.setLevel(level)
    
    return handler


def cleanup_old_logs(log_dir, days=LoggerConfig["LOG_RETENTION_DAYS"]):
    """
    Removes log files older than specified days from the given directory
    
    Args:
        log_dir (str): Directory containing log files
        days (int): Number of days to keep logs for
    """
    if not os.path.exists(log_dir):
        return
        
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # Get all log files in the directory
    log_files = glob.glob(os.path.join(log_dir, f"*{LoggerConfig['LOG_FILE_EXTENSION']}"))
    
    for log_file in log_files:
        try:
            # Extract date from filename (assuming format like component_YYYY-MM-DD.log)
            filename = os.path.basename(log_file)
            parts = filename.split('_')
            if len(parts) >= 2:
                date_str = parts[1].replace(LoggerConfig['LOG_FILE_EXTENSION'], '')
                file_date = datetime.strptime(date_str, LoggerConfig["DATE_FORMAT"])
                
                # Delete if older than cutoff
                if file_date < cutoff_date:
                    os.remove(log_file)
                    print(f"Removed old log file: {log_file}")
        except (ValueError, IndexError) as e:
            # If filename doesn't match expected format, skip it
            continue


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance that logs to both:
    - A component-specific dated log file (e.g., logs/Agent/Agent_2023-07-23.log)
    - A master dated log file (logs/all/all_2023-07-23.log)
    
    Logs older than configured days are automatically removed.
    """
    ensure_log_directory_exists(LoggerConfig["LOG_DIR"])

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(LoggerConfig["DEFAULT_LOG_LEVEL"])

    # 1. Component-specific file handler with date in filename
    component_handler = get_dated_log_handler(name, LoggerConfig["DEFAULT_LOG_LEVEL"])

    # 2. Master log file handler with date in filename
    master_handler = get_dated_log_handler("all", LoggerConfig["DEFAULT_LOG_LEVEL"])

    # Optional: also log to console
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(LoggerConfig["LOG_FORMAT"])
    console_handler.setFormatter(formatter)
    console_handler.setLevel(LoggerConfig["CONSOLE_LOG_LEVEL"])

    logger.addHandler(component_handler)
    logger.addHandler(master_handler)
    logger.addHandler(console_handler)

    return logger
