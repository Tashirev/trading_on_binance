from binance.spot import Spot
from config_read import *
from time import time

config_binance = config('binance')['trade']
coins_binance = ["BTC", "ETH", "BNB"]
pair_name =["BTCUSDT","ETHUSDT","BNBUSDT","BNBBTC","ETHBTC","BNBETH"]

client = Spot(key=config_binance['key'], secret=config_binance['secret'])

print(client)



def binance_read():

    #time_binance = client.time()['serverTime']
    # получение баланса по кошельку для "BTC", "ETH", "BNB"
    account = client.account()
    coins_balance = account['balances']
    coins = [coin for coin in coins_balance if coin['asset'] in coins_binance]
    # получение ticker_price
    ticker_price = [float(client.ticker_price(pair)['price']) for pair in pair_name]
    # получение depth
    #depth = [client.depth(pair, limit=5) for pair in pair_name]

    return coins, ticker_price#, depth
