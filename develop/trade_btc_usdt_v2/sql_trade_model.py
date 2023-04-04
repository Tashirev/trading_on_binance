import sys
sys.path.append('C:/Users/denis/binance')
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

class Trade_BTCUSDT(Base):
    __tablename__ = 'trade_btcusdt'
    id = Column(Integer, primary_key=True)
    local_time = Column(TIMESTAMP)
    period_time = Column(String)
    predict = Column(Integer)  # 0-6
    volume_status = Column(Integer)  # статус не превышения минимально объёма крипты в кошельке
    profit_fee_status = Column(Integer)  # статус превышения прибыли комиссии
    min_order_status = Column(Integer) # статус превышения минимально допустимого размера ордера Binance
    volume_before = Column(String) # btc+eth+bnb in usdt before trade
    fee_usdt = Column(String)
    part = Column(String)
    slip_n = Column(String)
    profit_fee_coef = Column(String)
    btc_usdt_min_trade_count = Column(String)  # число подряд выполненых одинаковых сделок min
    btc_usdt_max_trade_count = Column(String)  # число подряд выполненых одинаковых сделок max
    bnb_btc = Column(String) # ticker_price
    eth_btc = Column(String) # ticker_price
    bnb_eth = Column(String) # ticker_price
    btc_usdt = Column(String) # ticker_price
    eth_usdt = Column(String) # ticker_price
    bnb_usdt = Column(String) # ticker_price
    volume_btc = Column(String) # кошелёк на binance
    volume_usdt = Column(String) # кошелёк на binance
    volume_btc_buy = Column(String) # расчётные значения
    volume_usdt_buy = Column(String) # расчётные значения
    volume_btc_sell = Column(String) # расчётные значения
    volume_usdt_sell = Column(String) # расчётные значения
    order_id = Column(String) # результат ордера
    order_symbol = Column(String)  # результат ордера
    order_price = Column(String)  # результат ордера
    order_commission = Column(String)  # результат ордера
    order_commission_asset = Column(String)  # результат ордера

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

