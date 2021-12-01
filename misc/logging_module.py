import logging

logger = logging.getLogger("test")
handlers = []
handlers.append(logging.FileHandler(filename="test.log", mode='w', encoding='UTF-8'))  # write into the file
handlers.append(logging.StreamHandler())  # output stream on the terminal.
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for handler in handlers:
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG) # set the level of handler, alternatively it can set one by one
    logger.addHandler(handler)
logger.setLevel(logging.INFO)   # the whole level
