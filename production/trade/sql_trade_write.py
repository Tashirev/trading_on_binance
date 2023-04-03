from sqlalchemy.orm import sessionmaker
from sql_trade_model import *

Session = sessionmaker(bind = engine)
session = Session()



def sql_trade_write(local_time, period_time, predict, trade, volume_before_usdt,
                    volume_profit_usdt, fee_usdt, part, slip_n, bnb_btc, eth_btc,
                    bnb_eth, btc_usdt, eth_usdt, bnb_usdt, volume_btc, volume_eth,
                    volume_bnb):

    # получение id последней записи в таблице
    id = session.query(Trade_Old).order_by(Trade_Old.id.desc()).first()
    # проверка, что в таблице имеется запись, иначе назначить id 0.
    id = 0 if id == None else id.id
    id +=1
    trade_old = Trade_Old(id=id,
                  local_time=local_time,
                  period_time = period_time,
                  predict = predict,
                  trade = trade,
                  volume_before_usdt = volume_before_usdt,
                  volume_profit_usdt = volume_profit_usdt,
                  fee_usdt = fee_usdt,
                  part = part,
                  slip_n = slip_n,
                  bnb_btc = bnb_btc,
                  eth_btc = eth_btc,
                  bnb_eth = bnb_eth,
                  btc_usdt = btc_usdt,
                  eth_usdt = eth_usdt,
                  bnb_usdt = bnb_usdt,
                  volume_btc = volume_btc,
                  volume_eth = volume_eth,
                  volume_bnb = volume_bnb
                 )
    session.add(trade_old)
    session.commit()
    return 'SQL ok'