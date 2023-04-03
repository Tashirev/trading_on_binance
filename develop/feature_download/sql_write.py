# функции записи признаков с binance в базу данных

from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from sql_model import *
from datetime import datetime

Session = sessionmaker(bind = engine)
session = Session()

def last_ticker():
    return session.query(func.max(Ticker.id_ticker)).scalar()

def sql_write(ticker_count, time_binance, pairs):

    ticker = Ticker(id_ticker=ticker_count, time_server=time_binance)
    ticker_price = TickerPrice(id_ticker = ticker_count,
                               btc_usdt=pairs[0][1]['price'],
                               eth_usdt=pairs[1][1]['price'],
                               bnb_usdt=pairs[2][1]['price'],
                               bnb_btc=pairs[3][1]['price'],
                               eth_btc=pairs[4][1]['price'],
                               bnb_eth=pairs[5][1]['price']
                               )

    btc_usdt_24hr = BTC_USDT_24hr(id_ticker = ticker_count,
                                  price_change = pairs[0][2]['priceChange'],
                                  price_change_percent = pairs[0][2]['priceChangePercent'],
                                  weighted_avg_price = pairs[0][2]['weightedAvgPrice'],
                                  prev_close_price = pairs[0][2]['prevClosePrice'],
                                  last_price = pairs[0][2]['lastPrice'],
                                  last_qty = pairs[0][2]['lastQty'],
                                  bid_price = pairs[0][2]['bidPrice'],
                                  bid_qty = pairs[0][2]['bidQty'],
                                  ask_price = pairs[0][2]['askPrice'],
                                  ask_qty = pairs[0][2]['askQty'],
                                  open_price = pairs[0][2]['openPrice'],
                                  high_price = pairs[0][2]['highPrice'],
                                  low_price = pairs[0][2]['lowPrice'],
                                  volume = pairs[0][2]['volume'],
                                  quote_volume = pairs[0][2]['quoteVolume'],
                                  open_time = datetime.fromtimestamp(pairs[0][2]['openTime']/1000),
                                  close_time = datetime.fromtimestamp(pairs[0][2]['closeTime']/1000),
                                  first_id = pairs[0][2]['firstId'],
                                  last_id = pairs[0][2]['lastId'],
                                  count =  pairs[0][2]['count']
                                 )

    eth_usdt_24hr = ETH_USDT_24hr(id_ticker = ticker_count,
                                  price_change = pairs[1][2]['priceChange'],
                                  price_change_percent = pairs[1][2]['priceChangePercent'],
                                  weighted_avg_price = pairs[1][2]['weightedAvgPrice'],
                                  prev_close_price = pairs[1][2]['prevClosePrice'],
                                  last_price = pairs[1][2]['lastPrice'],
                                  last_qty = pairs[1][2]['lastQty'],
                                  bid_price = pairs[1][2]['bidPrice'],
                                  bid_qty = pairs[1][2]['bidQty'],
                                  ask_price = pairs[1][2]['askPrice'],
                                  ask_qty = pairs[1][2]['askQty'],
                                  open_price = pairs[1][2]['openPrice'],
                                  high_price = pairs[1][2]['highPrice'],
                                  low_price = pairs[1][2]['lowPrice'],
                                  volume = pairs[1][2]['volume'],
                                  quote_volume = pairs[1][2]['quoteVolume'],
                                  open_time = datetime.fromtimestamp(pairs[1][2]['openTime']/1000),
                                  close_time = datetime.fromtimestamp(pairs[1][2]['closeTime']/1000),
                                  first_id = pairs[1][2]['firstId'],
                                  last_id = pairs[1][2]['lastId'],
                                  count = pairs[1][2]['count']
                                 )

    bnb_usdt_24hr = BNB_USDT_24hr(id_ticker = ticker_count,
                                  price_change = pairs[2][2]['priceChange'],
                                  price_change_percent = pairs[2][2]['priceChangePercent'],
                                  weighted_avg_price = pairs[2][2]['weightedAvgPrice'],
                                  prev_close_price = pairs[2][2]['prevClosePrice'],
                                  last_price = pairs[2][2]['lastPrice'],
                                  last_qty = pairs[2][2]['lastQty'],
                                  bid_price = pairs[2][2]['bidPrice'],
                                  bid_qty = pairs[2][2]['bidQty'],
                                  ask_price = pairs[2][2]['askPrice'],
                                  ask_qty = pairs[2][2]['askQty'],
                                  open_price = pairs[2][2]['openPrice'],
                                  high_price = pairs[2][2]['highPrice'],
                                  low_price = pairs[2][2]['lowPrice'],
                                  volume = pairs[2][2]['volume'],
                                  quote_volume = pairs[2][2]['quoteVolume'],
                                  open_time = datetime.fromtimestamp(pairs[2][2]['openTime']/1000),
                                  close_time = datetime.fromtimestamp(pairs[2][2]['closeTime']/1000),
                                  first_id = pairs[2][2]['firstId'],
                                  last_id = pairs[2][2]['lastId'],
                                  count =  pairs[2][2]['count']
                                 )

    bnb_btc_24hr = BNB_BTC_24hr(id_ticker = ticker_count,
                                  price_change = pairs[3][2]['priceChange'],
                                  price_change_percent = pairs[3][2]['priceChangePercent'],
                                  weighted_avg_price = pairs[3][2]['weightedAvgPrice'],
                                  prev_close_price = pairs[3][2]['prevClosePrice'],
                                  last_price = pairs[3][2]['lastPrice'],
                                  last_qty = pairs[3][2]['lastQty'],
                                  bid_price = pairs[3][2]['bidPrice'],
                                  bid_qty = pairs[3][2]['bidQty'],
                                  ask_price = pairs[3][2]['askPrice'],
                                  ask_qty = pairs[3][2]['askQty'],
                                  open_price = pairs[3][2]['openPrice'],
                                  high_price = pairs[3][2]['highPrice'],
                                  low_price = pairs[3][2]['lowPrice'],
                                  volume = pairs[3][2]['volume'],
                                  quote_volume = pairs[3][2]['quoteVolume'],
                                  open_time = datetime.fromtimestamp(pairs[3][2]['openTime']/1000),
                                  close_time = datetime.fromtimestamp(pairs[3][2]['closeTime']/1000),
                                  first_id = pairs[3][2]['firstId'],
                                  last_id = pairs[3][2]['lastId'],
                                  count =  pairs[3][2]['count']
                                 )

    eth_btc_24hr = ETH_BTC_24hr(id_ticker = ticker_count,
                                  price_change = pairs[4][2]['priceChange'],
                                  price_change_percent = pairs[4][2]['priceChangePercent'],
                                  weighted_avg_price = pairs[4][2]['weightedAvgPrice'],
                                  prev_close_price = pairs[4][2]['prevClosePrice'],
                                  last_price = pairs[4][2]['lastPrice'],
                                  last_qty = pairs[4][2]['lastQty'],
                                  bid_price = pairs[4][2]['bidPrice'],
                                  bid_qty = pairs[4][2]['bidQty'],
                                  ask_price = pairs[4][2]['askPrice'],
                                  ask_qty = pairs[4][2]['askQty'],
                                  open_price = pairs[4][2]['openPrice'],
                                  high_price = pairs[4][2]['highPrice'],
                                  low_price = pairs[4][2]['lowPrice'],
                                  volume = pairs[4][2]['volume'],
                                  quote_volume = pairs[4][2]['quoteVolume'],
                                  open_time = datetime.fromtimestamp(pairs[4][2]['openTime']/1000),
                                  close_time = datetime.fromtimestamp(pairs[4][2]['closeTime']/1000),
                                  first_id = pairs[4][2]['firstId'],
                                  last_id = pairs[4][2]['lastId'],
                                  count =  pairs[4][2]['count']
                                 )

    bnb_eth_24hr = BNB_ETH_24hr(id_ticker = ticker_count,
                                  price_change = pairs[5][2]['priceChange'],
                                  price_change_percent = pairs[5][2]['priceChangePercent'],
                                  weighted_avg_price = pairs[5][2]['weightedAvgPrice'],
                                  prev_close_price = pairs[5][2]['prevClosePrice'],
                                  last_price = pairs[5][2]['lastPrice'],
                                  last_qty = pairs[5][2]['lastQty'],
                                  bid_price = pairs[5][2]['bidPrice'],
                                  bid_qty = pairs[5][2]['bidQty'],
                                  ask_price = pairs[5][2]['askPrice'],
                                  ask_qty = pairs[5][2]['askQty'],
                                  open_price = pairs[5][2]['openPrice'],
                                  high_price = pairs[5][2]['highPrice'],
                                  low_price = pairs[5][2]['lowPrice'],
                                  volume = pairs[5][2]['volume'],
                                  quote_volume = pairs[5][2]['quoteVolume'],
                                  open_time = datetime.fromtimestamp(pairs[5][2]['openTime']/1000),
                                  close_time = datetime.fromtimestamp(pairs[5][2]['closeTime']/1000),
                                  first_id = pairs[5][2]['firstId'],
                                  last_id = pairs[5][2]['lastId'],
                                  count =  pairs[5][2]['count']
                                 )

    btc_usdt_depth = BTC_USDT_Depth(id_ticker = ticker_count,
                                    bids = pairs[0][3]['bids'],
                                    asks = pairs[0][3]['asks']
                                   )
    eth_usdt_depth = ETH_USDT_Depth(id_ticker = ticker_count,
                                    bids = pairs[1][3]['bids'],
                                    asks = pairs[1][3]['asks']
                                   )
    bnb_usdt_depth = BNB_USDT_Depth(id_ticker = ticker_count,
                                    bids = pairs[2][3]['bids'],
                                    asks = pairs[2][3]['asks']
                                   )
    bnb_btc_depth = BNB_BTC_Depth(id_ticker = ticker_count,
                                  bids = pairs[3][3]['bids'],
                                  asks = pairs[3][3]['asks']
                                 )
    eth_btc_depth = ETH_BTC_Depth(id_ticker = ticker_count,
                                  bids = pairs[4][3]['bids'],
                                  asks = pairs[4][3]['asks']
                                 )
    bnb_eth_depth = BNB_ETH_Depth(id_ticker = ticker_count,
                                  bids = pairs[5][3]['bids'],
                                  asks = pairs[5][3]['asks']
                                 )

    session.add(ticker)
    session.add(ticker_price)
    session.add(btc_usdt_24hr)
    session.add(eth_usdt_24hr)
    session.add(bnb_usdt_24hr)
    session.add(bnb_btc_24hr)
    session.add(eth_btc_24hr)
    session.add(bnb_eth_24hr)
    session.add(btc_usdt_depth)
    session.add(eth_usdt_depth)
    session.add(bnb_usdt_depth)
    session.add(bnb_btc_depth)
    session.add(eth_btc_depth)
    session.add(bnb_eth_depth)
    session.commit()

    return 'SQL ok'