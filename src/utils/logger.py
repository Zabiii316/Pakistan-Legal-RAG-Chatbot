import logging


def setup_logging(log_file='app.log'):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])
    logging.info('Logging is set up.')


def log_exception(e):
    logging.error(f'An error occurred: {e}')


def log_info(message):
    logging.info(message)


def log_debug(message):
    logging.debug(message)