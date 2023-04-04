from sqlalchemy.orm import sessionmaker
from sql_wallet_model import *

Session = sessionmaker(bind = engine)
session = Session()

def sql_walet_write(server_time, coins_wallet, prices_usdt):

    # получение id последней записи в таблице
    id = session.query(Wallet).order_by(Wallet.id.desc()).first()
    # проверка, что в таблице имеется запись, иначе назначить id 0.
    id = 0 if id == None else id.id

    for coin_wallet,price_usdt in zip(coins_wallet,prices_usdt):
        id +=1
        wallet = Wallet(id=id,
                        server_time=server_time,
                        coin=coin_wallet['asset'],
                        quantity=coin_wallet['free'],
                        price_usdt=price_usdt['price']
                        )
        session.add(wallet)
        session.commit()
    return 'SQL ok'