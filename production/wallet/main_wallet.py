from binance_wallet import *
from sql_wallet_write import *
from datetime import datetime
import time

def main():
    print('Binance wallet to SQL')

    while True:
        # запрос данных из binance
        server_time, coins, prices_usdt = binance_wallet(coins_binance)

        server_time = datetime.fromtimestamp(server_time / 1000)

        # запись данных в БД binance
        sql_status = sql_walet_write(server_time, coins, prices_usdt)

        quantity_in_usdt = sum([float(coins[i]['free'])*float(prices_usdt[i]['price']) for i in range(3)])

        print(f'SQL status: {sql_status}, quantity in USDT: {quantity_in_usdt}, time: {server_time}')

        for i in range(98):
            time.sleep(3)


if __name__ == '__main__':
    main()






