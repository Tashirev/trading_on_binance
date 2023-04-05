from create_labels import *
from sql_labels_read_write import *

def main():

    print('Start create labels.')
    print('testing gitignore')

    pairs_name = ['bnb_btc', 'eth_btc', 'bnb_eth']

    # запрос данных из БД binance,
    # возвращает pandas таблицу ticker_price
    ticker_price_df = sql_pairs_raed(pairs_name)
    ticker_price_df = ticker_price_df.astype('float')

    # создание целевой функции,
    # на входе таблица pandas ticker_price,
    # возвращает таблицу pandas labels_df
    labels_df = create_labels(ticker_price_df)

    # запись данных в БД binance
    sql_status = sql_labels_write(labels_df)

    print(f'Finish create labels. Status: {sql_status}')

    doli = labels_df[['bnb_btc_min',
           'bnb_btc_max',
           'eth_btc_min',
           'eth_btc_max',
           'bnb_eth_min',
           'bnb_eth_max',
           'not_buy']].sum() * 100 / labels_df.shape[0]

    print(f"{doli}")



if __name__ == '__main__':
    main()
