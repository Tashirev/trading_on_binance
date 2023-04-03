from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.ext.declarative import declarative_base
from config_read import *

config_bin_dev = config('postgres')['binance']
engine = create_engine(f"postgresql+psycopg2://"
                       f"{config_bin_dev['user_name']}:"
                       f"{config_bin_dev['password']}@"
                       f"{config_bin_dev['server']}/"
                       f"{config_bin_dev['base_name']}")
Base = declarative_base()

class Labels(Base):
    __tablename__ = 'labels'
    id_ticker = Column(Integer, primary_key=True)
    label = Column(Integer)
    bnb_btc_min = Column(Integer)
    bnb_btc_max = Column(Integer)
    bnb_btc_min_max = Column(Integer)
    eth_btc_min = Column(Integer)
    eth_btc_max = Column(Integer)
    eth_btc_min_max = Column(Integer)
    bnb_eth_min = Column(Integer)
    bnb_eth_max = Column(Integer)
    bnb_eth_min_max = Column(Integer)
    not_buy = Column(Integer)

Base.metadata.create_all(engine)