from savant_rs.py.log import get_logger, init_logging

# run LOGLEVEL=info python python/utils/log.py
# or LOGLEVEL=debug python python/utils/log.py

init_logging()

logger = get_logger(__name__)

logger.debug("Hello, world!")
logger.info("Hello, world!")
