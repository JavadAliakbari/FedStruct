import os
import logging

_DEV_LOG_LEVEL = logging.DEBUG


def get_logger(name: str = 'root', log_on_file=False, save_path='./', append=False):

    log_level = os.getenv('LOG_LEVEL', _DEV_LOG_LEVEL)

    # create logger
    logger = logging.getLogger(name)
    # stop propagting to root logger
    logger.propagate = False
    logger.setLevel(log_level)

    # create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create console handler and set level to debug
    terminal_handler = logging.StreamHandler()
    terminal_handler.setFormatter(formatter)
    logger.addHandler(terminal_handler)

    if log_on_file:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # mode = 'a' if append else 'w+'
        file_handler = logging.FileHandler(
            filename=f'{save_path}result_{name}.log', mode='w+', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
