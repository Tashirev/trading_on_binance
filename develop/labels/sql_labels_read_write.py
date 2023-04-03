import pandas as pd
from sql_labels_model import engine

def sql_pairs_raed(columns):

    ticker_price_df = pd.read_sql_table(table_name='ticker_price',
                                        con=engine, schema='public', index_col='id_ticker', columns=columns)

    return ticker_price_df


def sql_labels_write(labels_df):

    labels_df.to_sql(name='labels', con=engine, schema='public',
                     if_exists='replace', index=True, index_label='id_ticker', chunksize=None)

    return 'SQL ok'
