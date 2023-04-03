from sqlalchemy import create_engine, Column, String, TIMESTAMP, Integer
from sqlalchemy.ext.declarative import declarative_base
from config_read import *

config_bin_dev = config('postgres')['binance']
engine = create_engine(f"postgresql+psycopg2://"
                       f"{config_bin_dev['user_name']}:"
                       f"{config_bin_dev['password']}@"
                       f"{config_bin_dev['server']}/"
                       f"{config_bin_dev['base_name']}")
Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trade'
    id = Column(Integer, primary_key=True)
    local_time = Column(TIMESTAMP)
    period_time = Column(String)
    predict = Column(Integer)  # 0-6
    volume_status = Column(Integer)  # статус не превышения минимально объёма крипты в кошельке
    profit_fee_status = Column(Integer)  # статус превышения прибыли комиссии
    min_order_status = Column(Integer) # статус превышения минимально допустимого размера ордера Binance
    volume_before = Column(String) # btc+eth+bnb in usdt before trade
    profit_usdt = Column(String) # profit in usdt (after - before)
    fee_usdt = Column(String)
    part = Column(String)
    slip_n = Column(String)
    profit_fee_coef = Column(String)
    bnb_btc = Column(String) # ticker_price
    eth_btc = Column(String) # ticker_price
    bnb_eth = Column(String) # ticker_price
    btc_usdt = Column(String) # ticker_price
    eth_usdt = Column(String) # ticker_price
    bnb_usdt = Column(String) # ticker_price
    volume_btc = Column(String) # кошелёк на binance
    volume_eth = Column(String) # кошелёк на binance
    volume_bnb = Column(String) # кошелёк на binance
    volume_btc_buy = Column(String) # расчётные значения
    volume_eth_buy = Column(String) # расчётные значения
    volume_bnb_buy = Column(String) # расчётные значения
    volume_btc_sell = Column(String) # расчётные значения
    volume_eth_sell = Column(String) # расчётные значения
    volume_bnb_sell = Column(String) # расчётные значения
    orderId = Column(String) # результат ордера
    orderSymbol = Column(String)  # результат ордера
    orderPrice = Column(String)  # результат ордера
    orderCommission = Column(String)  # результат ордера
    orderCommissionAsset = Column(String)  # результат ордера



Base.metadata.create_all(engine)

