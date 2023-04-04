import sys
sys.path.append('..')
from config_read import *

from sqlalchemy import create_engine, Column, String, TIMESTAMP, Integer
from sqlalchemy.ext.declarative import declarative_base

config_bin_dev = config('postgres')['binance']
engine = create_engine(f"postgresql+psycopg2://"
                       f"{config_bin_dev['user_name']}:"
                       f"{config_bin_dev['password']}@"
                       f"{config_bin_dev['server']}/"
                       f"{config_bin_dev['base_name']}")
Base = declarative_base()

class Wallet(Base):
    __tablename__ = 'wallet'
    id = Column(Integer, primary_key=True)
    server_time = Column(TIMESTAMP)
    coin = Column(String)
    quantity = Column(String)
    price_usdt = Column(String)
    last_trade_operation = Column(String)
    last_trade_coin = Column(String)
    last_trade_price = Column(String)


Base.metadata.create_all(engine)

