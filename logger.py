import logging

import sys


logFormatter = logging.Formatter(
    "%(asctime)s [%(levelname)-5.5s] [%(name)s:%(filename)s:%(lineno)s - %(funcName)s] %(message)s")


def add_log_file(target_logger, path):
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(logFormatter)
    target_logger.addHandler(file_handler)


# new style: logger object
def get_logger(*, files=(), log_to_console=True,
               name=None,
               level=logging.INFO):
    this_logger = logging.Logger("logger")
    this_logger.setLevel(level)
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logFormatter)
        this_logger.addHandler(console_handler)
    if isinstance(files, str):
        add_log_file(this_logger, files)
    else:
        assert isinstance(files, (list, tuple))
        for path in files:
            add_log_file(this_logger, path)
    if name is not None:
        this_logger.name = name
    return this_logger


# old style: global logger

logger = get_logger()


def log_to_file(path):
    add_log_file(logger, path)
