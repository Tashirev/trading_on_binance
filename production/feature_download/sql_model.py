# создание модели базы данных под запись признаков полученных с binance
import os
import sys
sys.path.insert(0, os.path.abspath("../../"))
#sys.path.insert(0, os.path.abspath("/home/denis/binance/"))
#sys.path.append('C:/Users/denis/binance')
from config_read import *

from sqlalchemy import create_engine, Column, Integer, String, TIMESTAMP, ARRAY, BigInteger
from sqlalchemy.ext.declarative import declarative_base

config_bin_dev = config('postgres')['binance']
engine = create_engine(f"postgresql+psycopg2://"
                       f"{config_bin_dev['user_name']}:"
                       f"{config_bin_dev['password']}@"
                       f"{config_bin_dev['server']}/"
                       f"{config_bin_dev['base_name']}")
Base = declarative_base()

class Ticker(Base):
    __tablename__ = 'ticker'
    id_ticker = Column(Integer, primary_key=True)
    time_server = Column(TIMESTAMP)

class TickerPrice(Base):
    __tablename__ = 'ticker_price'
    id_ticker = Column(Integer, primary_key=True)
    bnb_btc = Column(String)
    eth_btc = Column(String)
    bnb_eth = Column(String)
    btc_usdt = Column(String)
    eth_usdt = Column(String)
    bnb_usdt = Column(String)

class Pair24hr():
    __tablename__ = 'pair_24hr'
    id_ticker = Column(Integer, primary_key=True)
    price_change = Column(String)
    price_change_percent = Column(String)
    weighted_avg_price = Column(String)
    prev_close_price = Column(String)
    last_price = Column(String)
    last_qty = Column(String)
    bid_price = Column(String)
    bid_qty = Column(String)
    ask_price = Column(String)
    ask_qty = Column(String)
    open_price = Column(String)
    high_price = Column(String)
    low_price = Column(String)
    volume = Column(String)
    quote_volume = Column(String)
    open_time = Column(TIMESTAMP)
    close_time = Column(TIMESTAMP)
    first_id = Column(BigInteger)
    last_id = Column(BigInteger)
    count = Column(BigInteger)

class BNB_BTC_24hr(Base, Pair24hr):
    __tablename__ = 'bnb_btc_24hr'

class BNB_ETH_24hr(Base, Pair24hr):
    __tablename__ = 'bnb_eth_24hr'

class ETH_BTC_24hr(Base, Pair24hr):
    __tablename__ = 'eth_btc_24hr'

class BNB_USDT_24hr(Base, Pair24hr):
    __tablename__ = 'bnb_usdt_24hr'

class BTC_USDT_24hr(Base, Pair24hr):
    __tablename__ = 'btc_usdt_24hr'

class ETH_USDT_24hr(Base, Pair24hr):
    __tablename__ = 'eth_usdt_24hr'

class Depth():
    __tablename__ = 'depth'
    id_ticker = Column(Integer, primary_key=True)
    bids = Column(ARRAY(String))
    asks = Column(ARRAY(String))

class BNB_BTC_Depth(Base, Depth):
    __tablename__ = 'bnb_btc_depth'

class BNB_ETH_Depth(Base, Depth):
    __tablename__ = 'bnb_eth_depth'

class ETH_BTC_Depth(Base, Depth):
    __tablename__ = 'eth_btc_depth'

class BNB_USDT_Depth(Base, Depth):
    __tablename__ = 'bnb_usdt_depth'

class BTC_USDT_Depth(Base, Depth):
    __tablename__ = 'btc_usdt_depth'

class ETH_USDT_Depth(Base, Depth):
    __tablename__ = 'eth_usdt_depth'

Base.metadata.create_all(engine)
