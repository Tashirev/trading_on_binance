# функиця запроса признаков с API binance
import os
import sys
sys.path.insert(0, os.path.abspath("../../"))
#sys.path.insert(0, os.path.abspath("/home/denis/binance/"))
#sys.path.append('C:/Users/denis/binance')
from config_read import *
from binance.spot import Spot

config_binance = config('binance')['trade']
client = Spot(api_key=config_binance['key'], api_secret=config_binance['secret'])
#client = Spot(key=config_binance['key'], secret=config_binance['secret'])
pairs_binance = ["BTCUSDT","ETHUSDT","BNBUSDT","BNBBTC","ETHBTC","BNBETH"]
pairs_name = ['btc_usdt','eth_usdt','bnb_usdt','bnb_btc','eth_btc','bnb_eth']

def binance_download(pairs_binance, pairs_name):
    pairs = list()
    time_binance = client.time()['serverTime']
    for pair_binance,pair_name in zip(pairs_binance,pairs_name):
        pairs_binance_price = client.ticker_price(pair_binance)
        #pair_binance_depth = client.depth(pair_binance)
        pair_binance_depth = {'lastUpdateId': 32728977388, 'bids': [['0', '0']], 'asks': [['0', '0']]}
        #pair_binance_24hr = client.ticker_24hr(pair_binance)
        pair_binance_24hr = {'symbol': 'BTCUSDT', 'priceChange': '0', 'priceChangePercent': '0',
         'weightedAvgPrice': '0', 'prevClosePrice': '0', 'lastPrice': '0',
         'lastQty': '0', 'bidPrice': '0', 'bidQty': '0', 'askPrice': '0',
         'askQty': '0', 'openPrice': '0', 'highPrice': '0',
         'lowPrice': '0', 'volume': '0', 'quoteVolume': '0',
         'openTime': 1676314615975, 'closeTime': 1676401015975, 'firstId': 2693598034, 'lastId': 2701133372,
         'count': 7535339}
        pairs.append([pair_name, pairs_binance_price, pair_binance_24hr, pair_binance_depth])

    return time_binance, pairs