import contextlib
import logging
import os

from chalicelib.new.config.infra import envars
from chalicelib.new.config.infra.log import CFDI_TO_FIX


class DBFormatter(logging.Formatter):
    def format(self, record):
        record.pid = os.getpid()
        return logging.Formatter.format(self, record)


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, _NOTHING, DEFAULT = range(10)

LEVEL_COLOR_MAPPING = {
    logging.DEBUG: (BLUE, DEFAULT),
    logging.INFO: (GREEN, DEFAULT),
    logging.WARNING: (YELLOW, DEFAULT),
    logging.ERROR: (RED, DEFAULT),
    logging.CRITICAL: (WHITE, RED),
}
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
COLOR_PATTERN = "%s%s%%s%s" % (COLOR_SEQ, COLOR_SEQ, RESET_SEQ)


class ColoredFormatter(DBFormatter):
    def format(self, record):
        fg_color, bg_color = LEVEL_COLOR_MAPPING.get(record.levelno, (GREEN, DEFAULT))
        record.levelname = COLOR_PATTERN % (30 + fg_color, 40 + bg_color, record.levelname)
        return DBFormatter.format(self, record)


def is_a_tty(stream):
    return hasattr(stream, "fileno") and os.isatty(stream.fileno())


def addLoggingLevel(levelName, levelNum, methodName=None):
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError("{} already defined in logging module".format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError("{} already defined in logging module".format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError("{} already defined in logger class".format(methodName))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)  # pylint: disable=protected-access

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


def configure_logging(handler):
    log_format = "%(asctime)s %(pid)s %(levelname)s %(name)s: %(message)s"
    logging.getLogger("botocore").setLevel(logging.WARNING)
    formatter = ColoredFormatter(log_format)
    _logger = logging.getLogger()
    with contextlib.suppress(IndexError):
        _logger.handlers.pop()  # Remove default handler
    _logger.addHandler(handler)
    _logger.setLevel(envars.LOG_LEVEL)
    handler.setFormatter(formatter)


addLoggingLevel("CFDI_TO_FIX", CFDI_TO_FIX)


global_handler = logging.StreamHandler()
if (
    os.name == "posix"
    and isinstance(global_handler, logging.StreamHandler)
    and is_a_tty(global_handler.stream)
):
    configure_logging(global_handler)
