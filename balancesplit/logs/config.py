import os
import json
import logging
from bisect import bisect
from datetime import datetime
from logging import config 


from . import setLogger

class LogFileConfig:
    def __init__(self, path, logger, debug):
        self.path = path
        self.log = logger
        self.debug = debug

class LevelFilter(logging.Filter):
    """
    LoggingFilter used to filter one or more specific log levels messages
    """
    def __init__(self, level):
        self.__level = level

    def filter(self, record):
        return record.levelno in self.__level


# Adapted from https://stackoverflow.com/a/68154386
class LevelFormatter(logging.Formatter):
    """LoggingFormatter used to specifiy the formatting per level"""
    def __init__(self, formats: dict[int, str], **kwargs):
        super().__init__()

        if "fmt" in kwargs:
            raise ValueError(
                "Format string must be passed to level-surrogate formatters, "
                "not this one"
            )

        self.formats = sorted(
            (level, logging.Formatter(fmt, **kwargs)) for level, fmt in formats.items()
        )

    def format(self, record: logging.LogRecord) -> str:
        idx = bisect(self.formats, (record.levelno, ), hi=len(self.formats) - 1)
        level, formatter = self.formats[idx]
        return formatter.format(record)


def config_logger(log_file_path, debug=None, disable_existing_loggers=True):
    """
    Function to configure the logging.
    All info is saved in a simple format on the log file path.
    Debug entries are saved to a separate file if debug is True
    Debug and warning and above are save in a verbose format.
    Warning and above are also printed to std.out

    Args:
        log_file_path (str): Folder where all logs for this run are saved
        debug (bool): if true, debug messages are saved
        no_exist_log (bool): if true, existing loggers are disabled
    """
    debug_path = os.path.join(os.path.dirname(log_file_path), "debug.log")
    simple_format = "%(message)s"
    verbose_format = "[%(asctime)s] %(levelname)s [%(filename)s %(name)s %(funcName)s (%(lineno)d)]: %(message)s"  # noqa: E501

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": disable_existing_loggers,
        "formatters":
            {
                "simple_formatter": {
                    "format": simple_format
                },
                "verbose_formatter": {
                    "format": verbose_format
                },
                "bylevel_formatter":
                    {
                        "()": LevelFormatter,
                        "formats":
                            {
                                logging.DEBUG: verbose_format,
                                logging.INFO: simple_format,
                                logging.WARNING: verbose_format,
                            },
                    },
            },
        "filters": {
            "only_debug": {
                "()": LevelFilter,
                "level": [logging.DEBUG]
            }
        },
        "handlers":
            {
                "stream_handler":
                    {
                        "class": "logging.StreamHandler",
                        "formatter": "simple_formatter",
                        "level": "WARNING",
                    },
                "file_handler":
                    {
                        "class": "logging.FileHandler",
                        "formatter": "bylevel_formatter",
                        "filename": log_file_path,
                        "level": "INFO",
                    },
                "file_handler_debug":
                    {
                        "class": "logging.FileHandler",
                        "formatter": "bylevel_formatter",
                        "filename": debug_path,
                        "mode": "w",
                        "delay": True,
                        "filters": ["only_debug"],
                    },
            },
        "loggers":
            {
                None:
                    {
                        "handlers":
                            ["stream_handler", "file_handler", "file_handler_debug"]
                            if debug else ["stream_handler", "file_handler"],
                        "level":
                            "DEBUG",
                    }
            },
    }

    config.dictConfig(LOGGING_CONFIG)


def get_git_info():
    """
    Get information of the current git commit

    If the package is installed with pip, read detailed version extracted by setuptools_scm.
    Otherwise, use gitpython to get the information from the git repo.
    """

    import optisplit

    path = optisplit.__path__[0]
    logging.debug(f"Package path: {path}")

    from .._version import __version__

    info = __version__
    logging.info(f"Version info [from pip]: {info}")



def init_logfile(log, args=None):
    """
    Put some intial information in the logfile

    Args:
        log : Logging instance
        args (dict): Dictionary with all command line arguments
    """
    logging.info(f"Initialize GBMT log file: {log.root.handlers[1].baseFilename} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    get_git_info()
    if args:
        logging.info("Command line arguments:")
        for key, value in args.items():
            logging.info(f"{key}: {value}")
    logging.info("")

def enable_file_logger(
    log_folder: str,
    filename: str,
    debug: bool = False,
    log_name: str | None = None,
    init_data: dict | None = None,
    disable_existing_loggers: bool = False,
):
    """Enable file logging.

    Args:
        log_folder (str): path to the folder where the log file should be stored
        filename (str): name of the log file
        debug (bool): whether to enable debug logging. Defaults to False.
        log_name (str, optional): name of the logger. Defaults to None.
        init_data (dict, optional): initial data to be logged. Defaults to None.
        disable_existing_loggers (bool): whether to disable existing loggers.
    """
    # create log folder if it does not exist
    if log_folder != '':
        path = os.path.join(log_folder, filename)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
    else:
        path = filename

    # configure logging
    config_logger(path, debug, disable_existing_loggers=disable_existing_loggers)

    # get logger and init configuration
    log = logging.getLogger(filename) if not log_name else logging.getLogger(log_name)
    log.setLevel(logging.INFO)
    setLogger(log)
    settings = LogFileConfig(path, log, debug)

    # Begin log file
    init_logfile(log, init_data)

    return settings
