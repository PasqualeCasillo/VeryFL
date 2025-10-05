# config/log.py
import logging
from datetime import datetime
import os 

encoding = "utf-8"
level = logging.INFO  # Cambia a DEBUG per debugging profondo
format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

log_folder = "log"

def get_file_name():
    time = datetime.strftime(datetime.now(), '%Y_%m_%d_')
    suffix = ".log"
    order = 0
    while True:
        filename = os.path.join(log_folder, time + str(order) + suffix)
        if os.path.exists(filename):
            order += 1
            continue
        else: 
            return filename

def set_log_config(verbose=False):
    """
    Configura logging per VeryFL.
    
    Args:
        verbose (bool): Se True, usa DEBUG level. Se False, usa INFO.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Console handler (solo INFO/WARNING/ERROR)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler (tutto, anche DEBUG)
    file_handler = logging.FileHandler(get_file_name(), encoding=encoding)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Root logger configuration
    logging.basicConfig(
        level=log_level,
        handlers=[console_handler, file_handler]
    )
    
    # Silenzia logger esterni verbosi
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)