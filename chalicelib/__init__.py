import logging
import os


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


handler = logging.StreamHandler()
log_format = "%(asctime)s %(pid)s %(levelname)s %(name)s: %(message)s"
if os.name == "posix" and isinstance(handler, logging.StreamHandler) and is_a_tty(handler.stream):
    formatter = ColoredFormatter(log_format)
    _logger = logging.getLogger()
    _logger.handlers.pop()  # Remove default handler
    _logger.addHandler(handler)
    _logger.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
