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

class Trade_Old(Base):
    __tablename__ = 'trade_old'
    id = Column(Integer, primary_key=True)
    local_time = Column(TIMESTAMP)
    period_time = Column(String)
    predict = Column(Integer)  # 0-6
    trade = Column(Integer)    # 0 - no, 1 - yes
    volume_before_usdt = Column(String) # btc+eth+bnb in usdt before trade
    volume_profit_usdt = Column(String) # profit in usdt (after - before)
    fee_usdt = Column(String)
    part = Column(String)
    slip_n = Column(String)
    bnb_btc = Column(String) # ticker_price
    eth_btc = Column(String) # ticker_price
    bnb_eth = Column(String) # ticker_price
    btc_usdt = Column(String) # ticker_price
    eth_usdt = Column(String) # ticker_price
    bnb_usdt = Column(String) # ticker_price
    volume_btc = Column(String) # кошелёк на binance
    volume_eth = Column(String) # кошелёк на binance
    volume_bnb = Column(String) # кошелёк на binance


Base.metadata.create_all(engine)

