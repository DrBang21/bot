import logging


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("dj")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(name)s %(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logger()
