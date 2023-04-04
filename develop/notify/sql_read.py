import sys
sys.path.append('C:/Users/denis/binance')
from config_read import *

import pandas as pd
from sqlalchemy import create_engine

config_bin_dev = config('postgres')['binance']
engine = create_engine(f"postgresql+psycopg2://"
                       f"{config_bin_dev['user_name']}:"
                       f"{config_bin_dev['password']}@"
                       f"{config_bin_dev['server']}/"
                       f"{config_bin_dev['base_name']}")

def read_settings_sql(time_first, time_last):

    #  trade['orderSymbol','predict','volume_btc','volume_usdt','btc_usdt','volume_before']
    sql_request = f"SELECT id, local_time, order_symbol, predict, volume_btc, volume_usdt, volume_before, btc_usdt  FROM trade_btcusdt WHERE local_time BETWEEN '{time_first}' AND '{time_last}'" # 'SELECT orderSymbol, predict, volume_btc, volume_usdt, btc_usdt, volume_before  FROM trade_btcusdt ORDER BY id DESC LIMIT 10'
    trade = pd.read_sql(sql_request, engine)
    trade[['predict','volume_btc','volume_usdt','btc_usdt','volume_before']] = trade[['predict','volume_btc','volume_usdt','btc_usdt','volume_before']].astype('float')


    return trade