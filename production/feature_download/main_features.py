from binance_download import *
from sql_write import *
from datetime import datetime
from time import sleep

def main():
    print('Binance features to SQL')
    ticker_count = last_ticker() + 1 # должно быть значение послекднего тикера из БД ticker столбец id_ticker +1

    while True:
        # запрос данных из binance
        response = binance_download(pairs_binance, pairs_name)

        server_time = datetime.fromtimestamp(response[0] / 1000)
        # запись данных в БД binance
        sql_status = sql_write(ticker_count, server_time, response[1])
        # увеличение счётчика записей на 1
        ticker_count += 1
        sleep(0.5)
        print(f'SQL status: {sql_status}, ticker_count: {ticker_count}, time: {server_time}')


if __name__ == '__main__':
    main()






