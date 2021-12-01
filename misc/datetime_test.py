import datetime
import time

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
import logging

handler = logging.FileHandler(f"{time.strftime('%Y%m%d_%H', time.localtime())}", encoding='UTF-8')
