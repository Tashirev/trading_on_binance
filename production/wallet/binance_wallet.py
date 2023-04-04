import sys
sys.path.append('..')
from config_read import *

from binance.spot import Spot

config_binance = config('binance')['trade']
coins_binance = ["BTC", "ETH", "BNB"]
coins_name = ['btc', 'eth', 'bnb']

client = Spot(key=config_binance['key'], secret=config_binance['secret'])
account = client.account()

def binance_wallet(coins_binance):

    time_binance = client.time()['serverTime']
    coins_balance = account['balances']
    coins = [coin for coin in coins_balance if coin['asset'] in coins_binance]
    prices_usdt = [client.ticker_price(coin+'USDT') for coin in coins_binance]

    return time_binance, coins, prices_usdt
