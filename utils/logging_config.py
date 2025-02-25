from pathlib import Path
import logging
import os


def setup_logging(log_level=None, log_file=None):
    """
    Set up logging configuration for the application.

    Args:
        log_level (str, optional): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str, optional): Path to log file
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # Set default log level
    level = getattr(logging, log_level if log_level else "INFO")

    # Configure root logger
    logging_config = {
        "level": level,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }

    # Add file handler if specified
    if log_file:
        logging_config["filename"] = log_file
        logging_config["filemode"] = "a"  # Append mode

    # Apply configuration
    logging.basicConfig(**logging_config)

    # Return configured logger
    return logging.getLogger()
