import os
import logging

LOG_LEVEL = os.environ.get('RL_COPULA_POLICY_LOG_LEVEL', logging.DEBUG)
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

_log_handler = None
_initialised_names = set()
def init_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    global _log_handler
    if _log_handler is None:
        _log_handler = logging.StreamHandler()
        _log_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    global _initialised_names
    if name not in _initialised_names:
        logger.addHandler(_log_handler)
        _initialised_names.add(name)
    logger.propagate = False
