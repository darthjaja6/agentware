# import logging
import colorlog
import logging
import inspect

from agentware.singleton import Singleton


class CallerFormatter(colorlog.ColoredFormatter):
    def format(self, record):
        # Get the caller's frame
        caller_frame = inspect.stack()[9]
        # Get the caller's filename and line number
        caller_filename = caller_frame.filename
        caller_lineno = caller_frame.lineno

        # Update the record's values
        record.filename = caller_filename
        record.lineno = caller_lineno

        return super().format(record)


class Logger(metaclass=Singleton):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __init__(self):
        self.logger = colorlog.getLogger('logger')
        self.level = self.DEBUG
        formatter = CallerFormatter(
            '%(log_color)s%(asctime)s - %(levelname)s - [%(filename)s: %(lineno)d] %(message)s',
            log_colors={
                'DEBUG': 'reset',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red'
            },
            reset=True,
            style='%'
        )
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def set_level(self, level):
        if not level in [self.DEBUG, self.INFO, self.WARNING, self.ERROR, self.CRITICAL]:
            raise ValueError(f"log level {level} in valid")
        self.level = level

    def debug(self, message, bg_color=None):
        if self.level > self.DEBUG:
            return
        if bg_color:
            self.logger.debug(
                colorlog.escape_codes['bg_color'] + message + colorlog.escape_codes['reset'])
        else:
            self.logger.debug(message)

    def info(self, message):
        if self.level > self.INFO:
            return
        self.logger.info(message)

    def warning(self, message):
        if self.level > self.WARNING:
            return
        self.logger.warning(message)

    def error(self, message):
        if self.level > self.ERROR:
            return
        self.logger.error(message)

    def critical(self, message):
        if self.level > self.CRITICAL:
            return
        self.logger.critical(message)
