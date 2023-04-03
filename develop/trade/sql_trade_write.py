from sqlalchemy.orm import sessionmaker
from sql_trade_model import *

Session = sessionmaker(bind = engine)
session = Session()



def sql_trade_write(local_time, period_time, predict, volume_status, profit_fee_status,
                    min_order_status, volume_before, profit_usdt, fee_usdt, part,
                    slip_n, profit_fee_coef, bnb_btc, eth_btc,
                    bnb_eth, btc_usdt, eth_usdt, bnb_usdt, volume_btc, volume_eth,
                    volume_bnb,volume_btc_buy, volume_eth_buy, volume_bnb_buy,
                                         volume_btc_sell, volume_eth_sell, volume_bnb_sell,
                    orderId, orderSymbol, orderPrice, orderCommission, orderCommissionAsset):

    # получение id последней записи в таблице
    id = session.query(Trade).order_by(Trade.id.desc()).first()
    # проверка, что в таблице имеется запись, иначе назначить id 0.
    id = 0 if id == None else id.id
    id +=1
    trade = Trade(id=id,
                  local_time=local_time,
                  period_time = period_time,
                  predict = predict,
                  volume_status = volume_status,
                  profit_fee_status=profit_fee_status,
                  min_order_status=min_order_status,
                  volume_before = volume_before,
                  profit_usdt = profit_usdt,
                  fee_usdt = fee_usdt,
                  part = part,
                  slip_n = slip_n,
                  profit_fee_coef = profit_fee_coef,
                  bnb_btc = bnb_btc,
                  eth_btc = eth_btc,
                  bnb_eth = bnb_eth,
                  btc_usdt = btc_usdt,
                  eth_usdt = eth_usdt,
                  bnb_usdt = bnb_usdt,
                  volume_btc = volume_btc,
                  volume_eth = volume_eth,
                  volume_bnb = volume_bnb,
                  volume_btc_buy = volume_btc_buy,
                  volume_eth_buy = volume_eth_buy,
                  volume_bnb_buy = volume_bnb_buy,
                  volume_btc_sell = volume_btc_sell,
                  volume_eth_sell = volume_eth_sell,
                  volume_bnb_sell = volume_bnb_sell,
                  orderId = orderId,
                  orderSymbol = orderSymbol,
                  orderPrice = orderPrice,
                  orderCommission = orderCommission,
                  orderCommissionAsset = orderCommissionAsset
                 )
    session.add(trade)
    session.commit()
    return 'SQL ok'