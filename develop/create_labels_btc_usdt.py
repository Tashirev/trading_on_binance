import numpy as np
import pandas as pd
from tqdm import tqdm


pairs_name = ['btc_usdt']

# функция получения чередующихся максимумов и минимумов для пары, с учётом комиссии биржи
# df - таблица с trendom, pair_name - название пары, fee - размер комиссии за операцию на бирже (разница максимума и минимума должна быть больше стоимости умноженной на fee)
def create_min_max(df, pair_name, fee):
    df['index'] = df.index  # столбец индекса для получения среза по мин и макс

    # начальные значения переменных
    i_max = list()  # список индексов максимумов
    i_min = list()  # список индексов минимумов
    value_last_max = 0  # значение текущего максимума при движении по ряду
    value_last_min = 0  # значение текущего минимума при движении по ряду
    i_last_max = 0  # индекс текущего максимума при движении по ряду
    i_last_min = 0  # индекс текущего минимума при движении по ряду
    task = True  # текущее состояние задачи: True - поиск максимума, False - поиск минимума
    for i, value in tqdm(enumerate(df[pair_name])):

        # поиск максимума начинается после фиксации минимума
        if task:  # условие начала поиска максимума
            # если величина растёт
            if value > value_last_max:
                value_last_max = value  # назначается текущий максимум
                i_last_max = i  # назначается текущий индекс максимума
            else:  # величина не меняется или уменьшается
                if value_last_max - value > fee * value:  # просадка превысила комиссию (фиксируется максимум, назначается начальный минимум)
                    i_max.append(i_last_max)  # фиксируется максимум
                    value_last_min = value  # назначается начальный минимум
                    i_last_min = i  # назначается начальный индекс минимума
                    task = False

        # поиск минимума начинается после фиксации максимума
        if not task:  # условие начала поиска минимума
            # если величина уменьшается
            if value < value_last_min:
                value_last_min = value  # назначается текущий минимум
                i_last_min = i  # назначается текущий индекс минимума
            else:  # величина не меняется или увеличивается
                if value - value_last_min > fee * value:  # рост превысил комиссию (фиксируется минимум, назначается начальный максимум)
                    i_min.append(i_last_min)  # фиксируется минимум
                    value_last_max = value  # назначается начальный максимум
                    i_last_max = i  # назначается начальный индекс максимума
                    task = True

    i_max = np.array(i_max) + 1
    i_min = np.array(i_min) + 1
    pair_max = df.query('index in @i_max')
    pair_min = df.query('index in @i_min')
    pair_max[pair_name + '_max'] = 1
    pair_min[pair_name + '_min'] = 1
    pair_max = pair_max.drop(columns=[pair_name, 'index'])
    pair_min = pair_min.drop(columns=[pair_name, 'index'])
    df = df.merge(pair_min, how='left', on='id_ticker')
    df = df.merge(pair_max, how='left', on='id_ticker')
    df[[pair_name + '_min', pair_name + '_max']] = df[[pair_name + '_min', pair_name + '_max']].fillna(0).astype('int')
    df = df.drop(columns=['index'])
    df[[pair_name + '_min', pair_name + '_max']] = df[[pair_name + '_min', pair_name + '_max']].fillna(0).astype('int')

    return df[[pair_name + '_min', pair_name + '_max']]

# функция создания финальной разметки
def create_labels(df, FEE, SLIP_N):

    # разметка без учёта двойных сделок
    for pair_name in ['btc_usdt']:

        min_max = create_min_max(df, pair_name, FEE*SLIP_N)
        df = pd.concat((df, min_max), axis=1)

    # двойная сделка
    df['duble_buy'] = (df['btc_usdt_min'] +
                                 df['btc_usdt_max'])>1
    df['duble_buy'] = df['duble_buy'].astype('int')

    # профит каждой сделки
    for pair_name in ['btc_usdt']:
        df[pair_name + '_ratio'] = df[pair_name]/df[pair_name][1]
        df[pair_name + '_deal'] = df[pair_name + '_min'] + df[pair_name + '_max']
        df[pair_name + '_profit'] = abs(df[df[pair_name + '_deal'] == 1][pair_name + '_ratio'] - df[df[pair_name + '_deal'] == 1][pair_name + '_ratio'].shift(-1))
        df[pair_name + '_profit'] = df[pair_name + '_profit'].fillna(0)

    # # устранение двойных сделок
    # df['profit_max'] = df[df['duble_buy'] == 1][['bnb_btc_profit','eth_btc_profit','bnb_eth_profit']].idxmax(axis=1)
    # df.loc[df['profit_max'] == 'bnb_btc_profit',['eth_btc_min', 'eth_btc_max','bnb_eth_min','bnb_eth_max']] = 0
    # df.loc[df['profit_max'] == 'eth_btc_profit',['bnb_btc_min', 'bnb_btc_max','bnb_eth_min','bnb_eth_max']] = 0
    # df.loc[df['profit_max'] == 'bnb_eth_profit',['bnb_btc_min', 'bnb_btc_max','eth_btc_min','eth_btc_max']] = 0

    # отсутствие сделки
    df['not_buy'] = (df['btc_usdt_min'] +
                               df['btc_usdt_max'])
    df['not_buy'] = (df['not_buy'] == 0).astype('int')

    # min_max - для удобства визуализации разметки
    for pair_name in ['btc_usdt']:
        df[pair_name + '_min_max'] = df[pair_name + '_min']*-1 + df[pair_name + '_max']

    # создание labels
    labels = df.copy()
    labels=labels.rename(
        columns={
            'not_buy':'0',
            'btc_usdt_min':'1',
            'btc_usdt_max':'2'
        }
    )
    labels = labels[['0','1','2']]
    label = labels.idxmax(axis=1).rename('label').astype('int')
    df = pd.concat((df, label), axis=1)

    return df[['btc_usdt_min', 'btc_usdt_max', 'btc_usdt_min_max',
               'not_buy', 'label']]



