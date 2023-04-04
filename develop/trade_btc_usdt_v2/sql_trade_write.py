import pandas as pd

from sqlalchemy.orm import sessionmaker
from sql_trade_model import *

Session = sessionmaker(bind = engine)
session = Session()

def sql_trade_write(local_time, period_time, predict, volume_status, profit_fee_status,
                    min_order_status, volume_before, fee_usdt, part,
                    slip_n, profit_fee_coef, btc_usdt_min_trade_count,
                    btc_usdt_max_trade_count, bnb_btc, eth_btc,
                    bnb_eth, btc_usdt, eth_usdt, bnb_usdt, volume_btc, volume_usdt,
                    volume_btc_buy, volume_usdt_buy, volume_btc_sell, volume_usdt_sell,
                    order_id, order_symbol, order_price, order_commission, order_commission_asset):

    # получение id последней записи в таблице
    id = session.query(Trade_BTCUSDT).order_by(Trade_BTCUSDT.id.desc()).first()
    # проверка, что в таблице имеется запись, иначе назначить id 0.
    id = 0 if id == None else id.id
    id +=1
    trade_btcusdt = Trade_BTCUSDT(id=id,
                  local_time=local_time,
                  period_time = period_time,
                  predict = predict,
                  volume_status = volume_status,
                  profit_fee_status=profit_fee_status,
                  min_order_status=min_order_status,
                  volume_before = volume_before,
                  fee_usdt = fee_usdt,
                  part = part,
                  slip_n = slip_n,
                  profit_fee_coef = profit_fee_coef,
                  btc_usdt_min_trade_count = btc_usdt_min_trade_count,
                  btc_usdt_max_trade_count = btc_usdt_max_trade_count,
                  bnb_btc = bnb_btc,
                  eth_btc = eth_btc,
                  bnb_eth = bnb_eth,
                  btc_usdt = btc_usdt,
                  eth_usdt = eth_usdt,
                  bnb_usdt = bnb_usdt,
                  volume_btc = volume_btc,
                  volume_usdt = volume_usdt,
                  volume_btc_buy = volume_btc_buy,
                  volume_usdt_buy = volume_usdt_buy,
                  volume_btc_sell = volume_btc_sell,
                  volume_usdt_sell = volume_usdt_sell,
                  order_id = order_id,
                  order_symbol = order_symbol,
                  order_price = order_price,
                  order_commission = order_commission,
                  order_commission_asset = order_commission_asset
                 )
    session.add(trade_btcusdt)
    session.commit()
    return 'SQL ok'

def read_settings_sql():

    settings = pd.read_sql('SELECT * FROM trading_settings ORDER BY id DESC LIMIT 1', engine)
    settings_time = settings['local_time'][0]
    trading_on = int(settings['trading_on'][0])
    start = int(settings['start'][0])
    part = float(settings['part'][0])
    fee = float(settings['fee'][0])
    profit_fee_coef = float(settings['profit_fee_coef'][0])
    trade_move_coef = float(settings['trade_move_coef'][0])
    centralization = float(settings['centralization'][0])

    return trading_on, start, part, fee, profit_fee_coef, trade_move_coef, centralization

def sql_trading_settings_write(local_time, trading_on, start, part, fee, profit_fee_coef, trade_move_coef, centralization):

    # получение id последней записи в таблице
    id = session.query(Trading_Settings).order_by(Trading_Settings.id.desc()).first()
    # проверка, что в таблице имеется запись, иначе назначить id 0.
    id = 0 if id == None else id.id
    id +=1
    trading_settings = Trading_Settings(id=id,
                                        local_time=local_time,
                                        trading_on=trading_on,
                                        start=start,
                                        part=part,
                                        fee=fee,
                                        profit_fee_coef=profit_fee_coef,
                                        trade_move_coef=trade_move_coef,
                                        centralization=centralization
                 )
    session.add(trading_settings)
    session.commit()
    return 'SQL ok'