import os
import sys
import logging
import time

from binance_download import *
from sql_write import *
from datetime import datetime
from time import sleep, ctime


def main():
    # создание логера вывода в stdout
    loger_stdout = logging.getLogger('loger_stdout')
    loger_stdout.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    loger_stdout.addHandler(stdout_handler)

    # создания логера записи в файл
    loger_file = logging.getLogger('loger_file')
    loger_file.setLevel(logging.INFO)
    file_handler = logging.FileHandler("logs/feature_download.log", "a")
    loger_file.addHandler(file_handler)

    loger_file.info(f'***** Binance features to SQL start {ctime()}')
    loger_stdout.info(f'***** Binance features to SQL start {ctime()}')
    ticker_count = last_ticker() + 1 # должно быть значение послекднего тикера из БД ticker столбец id_ticker +1


    while True:
        try:
            # запрос данных из binance
            response = binance_download(pairs_binance, pairs_name)

            server_time = datetime.fromtimestamp(response[0] / 1000)
            # запись данных в БД binance
            sql_status = sql_write(ticker_count, server_time, response[1])
            # увеличение счётчика записей на 1
            ticker_count += 1
            sleep(0.5)
            loger_stdout.info(f'SQL status: {sql_status}, ticker_count: {ticker_count}, time: {server_time}')
        except Exception as exc:
            loger_stdout.info(f'exc')

if __name__ == '__main__':
    main()






