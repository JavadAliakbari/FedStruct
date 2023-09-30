import os
import logging

_DEV_LOG_LEVEL = logging.DEBUG


def get_logger(
    name: str = "root",
    terminal=True,
    log_on_file=False,
    save_path="./",
    append=False,
):
    log_level = os.getenv("LOG_LEVEL", _DEV_LOG_LEVEL)

    # create logger
    logger = logging.getLogger(name)
    # stop propagting to root logger
    logger.propagate = False
    logger.setLevel(log_level)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # create console handler and set level to debug
    if terminal:
        terminal_handler = logging.StreamHandler()
        terminal_handler.setFormatter(formatter)
        logger.addHandler(terminal_handler)

    if log_on_file:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        filename = f"{save_path}result_{name}"
        i = ""
        while os.path.exists(f"{filename}{i}.log"):
            if i == "":
                i = 1
            else:
                i += 1

        # mode = 'a' if append else 'w+'
        file_handler = logging.FileHandler(
            filename=f"{filename}{i}.log", mode="w+", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
