import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sql_trading_settings_write import *
from datetime import datetime
from time import time, sleep

# отправка сообщения в телеграм
def send_telegram(text, token, chat_id):

    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    params = {
        "chat_id": chat_id,
        "text": text
        }
    url_req = f'https://api.telegram.org/bot{token}/sendMessage'
    results = session.get(url_req, params=params)
    #print('Отправка в телеграм:', results.json()[0]['result']['text'])
    print('Send to telegram:', results.json()['result']['text'])

# получение последней записи из телеграм
def read_telegram(token):

    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    url_req = f'https://api.telegram.org/bot{token}/getUpdates'
    results = session.get(url_req)

    try:
        message_id = results.json()['result'][-1]['message']['message_id']
    except:
        message_id = results.json()['result'][-1]['edited_message']['message_id']

    try:
        text = results.json()['result'][-1]['message']['text'].split(' ', 2)
    except:
        text = results.json()['result'][-1]['edited_message']['text'].split(' ', 2)

    # проверка корректности введённых данных 3 требования (для изменения значений): длинна = 2, текст соответствует списку, второе значение число
    if len(text) == 1 and text[0] == 'help':
        telegram_read_word = text[0]
        telegram_read_number = 999

    elif len(text) == 2:
        requirement_word = text[0] in ['trading_on','start','part','fee','profit_fee_coef','trade_move_coef','centralization']
        try:
            number = float(text[1])
        except ValueError:
            number = False
        requirement_number = type(number) is float
        if requirement_word and requirement_number:
            telegram_read_word = text[0]
            telegram_read_number = number
        else:
            telegram_read_word = None
            telegram_read_number = 999
    else:
        telegram_read_word = None
        telegram_read_number = 999



    return telegram_read_word, telegram_read_number, message_id


def read_settings_sql(engine):

    settings = pd.read_sql('SELECT * FROM trading_settings ORDER BY id DESC LIMIT 1', engine)
    settings_time = settings['local_time'][0]
    trading_on = int(settings['trading_on'][0])
    start = int(settings['start'][0])
    part = settings['part'][0]
    fee = settings['fee'][0]
    profit_fee_coef = settings['profit_fee_coef'][0]
    trade_move_coef = settings['trade_move_coef'][0]
    centralization = settings['centralization'][0]

    return settings_time, trading_on, start, part, fee, profit_fee_coef, trade_move_coef, centralization


def main():
    print('Telegram bot')
    config_telegram = config('telegram')
    telegram_token = config_telegram['token']
    chat_id = config_telegram['chat_id']
    message_id_last = None
    settings_time_last, trading_on_last, start_last, part_last, fee_last, profit_fee_coef_last, trade_move_coef_last, centralization_last = read_settings_sql(engine)

    while True:

        local_time = datetime.fromtimestamp(time())

        # чтение последней записи из telegram
        telegram_read_word, telegram_read_number, message_id = read_telegram(telegram_token)
        # при расхождении с текущими значениями запись в БД новых значений из телеграм
        if telegram_read_word == 'trading_on' and telegram_read_number in [0,1] and telegram_read_number != float(trading_on_last) and  message_id != message_id_last:
            sql_trading_settings_write(local_time, telegram_read_number, start_last, part_last, fee_last,
                                       profit_fee_coef_last, trade_move_coef_last, centralization_last)
        elif telegram_read_word == 'start' and telegram_read_number in [0,1] and telegram_read_number != float(start_last) and  message_id != message_id_last:
            sql_trading_settings_write(local_time, trading_on_last, telegram_read_number, part_last, fee_last,
                                       profit_fee_coef_last, trade_move_coef_last, centralization_last)
        elif telegram_read_word == 'part' and telegram_read_number != float(part_last) and  message_id != message_id_last:
            sql_trading_settings_write(local_time, trading_on_last, start_last, telegram_read_number, fee_last,
                                       profit_fee_coef_last, trade_move_coef_last, centralization_last)
        elif telegram_read_word == 'fee' and telegram_read_number != float(fee_last) and  message_id != message_id_last:
            sql_trading_settings_write(local_time, trading_on_last, start_last, part_last, telegram_read_number,
                                       profit_fee_coef_last, trade_move_coef_last, centralization_last)
        elif telegram_read_word == 'profit_fee_coef' and telegram_read_number != float(profit_fee_coef_last) and  message_id != message_id_last:
            sql_trading_settings_write(local_time, trading_on_last, start_last, part_last, fee_last,
                                       telegram_read_number, trade_move_coef_last, centralization_last)
        elif telegram_read_word == 'trade_move_coef' and telegram_read_number != float(trade_move_coef_last) and  message_id != message_id_last:
            sql_trading_settings_write(local_time, trading_on_last, start_last, part_last, fee_last,
                                       profit_fee_coef_last, telegram_read_number, centralization_last)
        elif telegram_read_word == 'centralization' and telegram_read_number != float(centralization_last) and  message_id != message_id_last:
            sql_trading_settings_write(local_time, trading_on_last, start_last, part_last, fee_last,
                                       profit_fee_coef_last, trade_move_coef_last,telegram_read_number)

        # вывод инструкции в телеграм при слове 'help'
        if telegram_read_word == 'help' and message_id != message_id_last:
            settings = 'Доступны следующие команды:\n trading_on (0 - торговля разрешена, 1 - запрещена)\n' \
                       ' start (0 - не обновлять начальные значения, 1 - обновить начальные значения)\n' \
                       ' part (коэффициент продоваемого объёма)\n fee (размер комиссии binance)\n' \
                       ' profit_fee_coef (коэффициент превышения прибыли над комиссией)\n' \
                       ' trade_move_coef (коэффициент защиты от резких изменений курса)\n' \
                       ' centralization (коэффициент ассимитрии объёма для движения к центру)\n' \
                       'Текущие значения:\n' \
                       f' trading_on = {trading_on_last}\n' \
                       f' start = {start_last}\n' \
                       f' part = {part_last}\n' \
                       f' fee = {fee_last}\n' \
                       f' profit_fee_coef = {profit_fee_coef_last}\n' \
                       f' trade_move_coef = {trade_move_coef_last}\n' \
                       f' centralization = {centralization_last}\n'
            send_telegram(settings, telegram_token, chat_id)

        # чтение последней записи из БД
        settings_time, trading_on, start, part, fee, profit_fee_coef, trade_move_coef, centralization = read_settings_sql(engine)

        if trading_on_last != trading_on:  # отправка сообщения в случае изменения trading_on
            message = f'{str(settings_time)} change trading_on {trading_on}'
            send_telegram(message, telegram_token, chat_id)
            trading_on_last = trading_on

        if start_last != start:  # отправка сообщения в случае изменения trading_on
            message = f'{str(settings_time)} change start {start}'
            send_telegram(message, telegram_token, chat_id)
            start_last = start

        if part_last != part:  # отправка сообщения в случае изменения part
            message = f'{str(settings_time)} change part {part}'
            send_telegram(message, telegram_token, chat_id)
            part_last = part

        if fee_last != fee:  # отправка сообщения в случае изменения fee
            message = f'{str(settings_time)} change fee {fee}'
            send_telegram(message, telegram_token, chat_id)
            fee_last = fee

        if profit_fee_coef_last != profit_fee_coef:  # отправка сообщения в случае изменения profit_fee_coef
            message = f'{str(settings_time)} change profit_fee_coef {profit_fee_coef}'
            send_telegram(message, telegram_token, chat_id)
            profit_fee_coef_last = profit_fee_coef

        if trade_move_coef_last != trade_move_coef:  # отправка сообщения в случае изменения trade_move_coef
            message = f'{str(settings_time)} change trade_move_coef {trade_move_coef}'
            send_telegram(message, telegram_token, chat_id)
            trade_move_coef_last = trade_move_coef

        if centralization_last != centralization:  # отправка сообщения в случае изменения trade_move_coef
            message = f'{str(settings_time)} change centralization {centralization}'
            send_telegram(message, telegram_token, chat_id)
            centralization_last = centralization

        print(f'Bot is working, time: {local_time}, trading_on: {trading_on_last}, start: {start_last}'
              f' part: {part_last}, fee: {fee_last}, profit_fee_coef: {profit_fee_coef_last},'
              f' trade_move_coef: {trade_move_coef_last}, centralization: {centralization_last}')

        message_id_last = message_id

        sleep(3)

if __name__ == '__main__':
    main()