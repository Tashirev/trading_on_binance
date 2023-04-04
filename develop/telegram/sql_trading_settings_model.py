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

class Trading_Settings(Base):
    __tablename__ = 'trading_settings'
    id = Column(Integer, primary_key=True)
    local_time = Column(TIMESTAMP)
    trading_on = Column(Integer)
    start = Column(Integer)
    part = Column(String)
    fee = Column(String)
    profit_fee_coef = Column(String)
    trade_move_coef = Column(String)
    centralization = Column(String)

Base.metadata.create_all(engine)

