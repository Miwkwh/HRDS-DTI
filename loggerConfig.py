import logging
import sys
from datetime import datetime
import os

class LoggerConfig:
    def __init__(self):
        pass

    @staticmethod
    def setup_logger(
        name: str = "default_logger",
        level: int = logging.INFO,
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_prefix: str = "log",
        log_dir: str = "./log",  
        sub_dir: str = None,  
        file_mode: str = "w",
    ) -> logging.Logger:

        logger = logging.getLogger(name)
        logger.setLevel(level)

        formatter = logging.Formatter(log_format)

    
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{log_prefix}_{timestamp}.log"
        full_log_dir = log_dir
        if sub_dir:
            full_log_dir = os.path.join(log_dir, sub_dir)
        os.makedirs(full_log_dir, exist_ok=True)
        log_filepath = os.path.join(full_log_dir, log_filename)

        fh = logging.FileHandler(log_filepath, mode=file_mode, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger