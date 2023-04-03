from sqlalchemy.orm import sessionmaker
from sql_trading_settings_model import *

Session = sessionmaker(bind = engine)
session = Session()


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