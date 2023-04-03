import numpy as np

def calculation(trade): # trade['orderSymbol','predict','volume_btc','volume_usdt','btc_usdt','volume_before']

    trade_graf = trade[['predict','volume_before','btc_usdt','volume_usdt']]
    trade['volume_not_trade'] = (trade.loc[0,'volume_btc'] * trade.loc[:,'btc_usdt'] +
                                 trade.loc[0,'volume_usdt'])
    trade_graf['profit'] = (trade.loc[:,'volume_before'] - trade.loc[:,'volume_not_trade'])
    trade_graf['volume_btc_usdt'] = trade.loc[:,'volume_btc'] * trade.loc[:,'btc_usdt']
    trade_graf['trade'] = (trade.loc[:,'order_symbol'] != 'None').astype('int')
    trade_graf['trade_on_profit'] = trade_graf.loc[:,'trade'] * trade_graf.loc[:,'profit']
    trade_graf.loc[trade_graf['trade_on_profit'] == 0,'trade_on_profit'] = np.nan

    # наложение predict на trade
    trade_graf['predict_min_on_btc_usdt'] = trade.loc[:,'btc_usdt'] * (trade.loc[:,'predict'] == 1)  # заменить 0 на nan
    trade_graf['predict_max_on_btc_usdt'] = trade.loc[:,'btc_usdt'] * (trade.loc[:,'predict'] == 2)  # заменить 0 на nan
    trade_graf['predict_min_on_btc_usdt'].replace(0, np.nan, inplace=True)
    trade_graf['predict_max_on_btc_usdt'].replace(0, np.nan, inplace=True)

    # наложение сделок на trade
    trade_graf['trade_min_on_btc_usdt'] = trade_graf.loc[:,'trade'] * trade.loc[:,'btc_usdt'] * (trade.loc[:,'predict'] == 1)  # заменить 0 на nan
    trade_graf['trade_max_on_btc_usdt'] = trade_graf.loc[:,'trade'] * trade.loc[:,'btc_usdt'] * (trade.loc[:,'predict'] == 2)  # заменить 0 на nan
    trade_graf['trade_min_on_btc_usdt'].replace(0, np.nan, inplace=True)
    trade_graf['trade_max_on_btc_usdt'].replace(0, np.nan, inplace=True)

    return trade_graf