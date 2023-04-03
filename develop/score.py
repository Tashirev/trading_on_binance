import pandas as pd
import numpy as np


def sell_buy(volume_x, volume_y, x_y_price, x_usdt_price, y_usdt_price, part, FEE, SLIP_N):  # продаваемая валюта X, покупаемая валюта Y, ....
    volume_x_sell = volume_x * part  # продаваемый объём
    volume_y_buy = volume_x_sell * y_usdt_price * x_y_price / x_usdt_price  # покупаемый объём
    fee = volume_y_buy * FEE * SLIP_N  # размер комиссии


    if True:#(volume_y_buy - volume_x_sell) > fee * 1.5:# and volume_y_buy > 10:  # если прибыль сделки превышает комиссию и объём покупки превышает 10 usdt, то совершить сделку
        volume_x = volume_x - volume_x_sell  # объём продаваемой валюты после сделки
        volume_y = volume_y + volume_y_buy - fee  # объём покупаемой валюты после сделки
        status = 1
    else:
        status = 0
        fee = 0
    return volume_x, volume_y, fee, status

def sell_buy_not_fee(volume_x, volume_y, x_y_price, x_usdt_price, y_usdt_price, part, FEE):  # продаваемая валюта X, покупаемая валюта Y, ....

    min_order_status = 0

    volume_x_sell = volume_x * part  # продаваемый объём
    volume_y_buy = volume_x_sell * y_usdt_price * x_y_price / x_usdt_price  # покупаемый объём
    fee = volume_y_buy * FEE  # размер комиссии

    if volume_x_sell > 11:  # статус превышения минимально допустимого ордера Binance 10 usdt
        min_order_status = 1

    if min_order_status == 1:  # выполнить сделку, если статус 1
        volume_x = volume_x - volume_x_sell  # объём продаваемой валюты после сделки
        volume_y = volume_y + volume_y_buy  # объём покупаемой валюты после сделки
        status = 1
    else:
        status = 0
        fee = 0
    return volume_x, volume_y, fee, status


def profit_status(price_max, price_min, fee, n, trade_count):
    status_buy = (price_max - price_min)/price_max > n*fee*(trade_count+1)
    return status_buy

def trade(test, predict, FEE, SLIP_N):
    volume_list = list()
    volume_btc_list = list()
    volume_eth_list = list()
    volume_bnb_list = list()
    volume_not_trade_list = list()

    part = 0.2  #доля для продажи

    volume_btc = 100
    volume_eth = 100
    volume_bnb = 100
    volume_start = volume_btc + volume_eth + volume_bnb
    fee_usdt = 0
    trade_count = 0

    # Для определения профита с покупки (price_sell - price_buy)/ price_sell - profit_fee_coef * fee)
    # нужно сохранять цену последней сделки по каждой паре
    bnb_btc_last_trade = test[0,3]
    eth_btc_last_trade = test[0, 4]
    bnb_eth_last_trade = test[0, 5]
    profit_fee_coef = 50


    for i, pred in enumerate(predict):

        if i!=0:
            if pred == 1 and profit_status(bnb_btc_last_trade, test[i,3], FEE, profit_fee_coef):# and last_trade != 'btc_bnb':   # продажа btc, покупка bnb
                volume_btc, volume_bnb, fee, status = sell_buy(volume_btc, volume_bnb, 1/test[i,3], test[i,0], test[i,2], part, FEE,
                                                  SLIP_N)
                trade_count += status

                volume_eth = volume_eth * (test[i,1]/test[i-1,1])

            elif pred == 2 and volume_bnb/(volume_btc + volume_eth + volume_bnb) > 0.1  and profit_status(test[i,3], bnb_btc_last_trade, FEE, profit_fee_coef):# and last_trade != 'bnb_btc':   # продажа bnb, покупка btc
                volume_bnb, volume_btc, fee, status = sell_buy(volume_bnb, volume_btc, test[i, 3], test[i, 2], test[i, 0], part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_eth = volume_eth * (test[i, 1] / test[i - 1, 1])

            elif pred == 3 and profit_status(eth_btc_last_trade, test[i,4], FEE, profit_fee_coef):# and last_trade != 'btc_eth':   # продажа btc, покупка eth
                volume_btc, volume_eth, fee, status = sell_buy(volume_btc, volume_eth, 1/test[i, 4], test[i, 0], test[i, 1], part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_bnb = volume_bnb * (test[i, 2] / test[i - 1, 2])

            elif pred == 4  and profit_status(test[i,4], eth_btc_last_trade, FEE, profit_fee_coef):# and last_trade != 'eth_btc':   # продажа eth, покупка btc
                volume_eth, volume_btc, fee, status = sell_buy(volume_eth, volume_btc, test[i, 4], test[i, 1], test[i, 0], part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_bnb = volume_bnb * (test[i, 2] / test[i - 1, 2])

            elif pred == 5 and profit_status(bnb_eth_last_trade, test[i,5], FEE, profit_fee_coef):# and last_trade != 'eth_bnb':   # продажа eth, покупка bnb
                volume_eth, volume_bnb, fee, status = sell_buy(volume_eth, volume_bnb, 1/test[i, 5], test[i, 1], test[i, 2], part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_btc = volume_btc * (test[i, 0] / test[i - 1, 0])

            elif pred == 6 and volume_bnb/(volume_btc + volume_eth + volume_bnb) > 0.1 and profit_status(test[i,5], bnb_eth_last_trade, FEE, profit_fee_coef):# and last_trade != 'bnb_eth':   # продажа bnb, покупка eth
                volume_bnb, volume_eth, fee, status = sell_buy(volume_bnb, volume_eth, test[i, 5], test[i, 2], test[i, 1], part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_btc = volume_btc * (test[i, 0] / test[i - 1, 0])

            else:
                fee = 0
                volume_btc = volume_btc * (test[i, 0] / test[i - 1, 0])
                volume_eth = volume_eth * (test[i, 1] / test[i - 1, 1])
                volume_bnb = volume_bnb * (test[i, 2] / test[i - 1, 2])
        else:
            fee = 0
        fee_usdt += fee
        volume_not_trade = (100 * (1 + (test[i, 0] - test[0, 0]) / test[0, 0]) +
                            100 * (1 + (test[i, 1] - test[0, 1]) / test[0, 1]) +
                            100 * (1 + (test[i, 2] - test[0, 2]) / test[0, 2])
                            )

        volume_list.append(volume_btc + volume_eth + volume_bnb)
        volume_btc_list.append(volume_btc)
        volume_eth_list.append(volume_eth)
        volume_bnb_list.append(volume_bnb)
        volume_not_trade_list.append(volume_not_trade)

    return volume_list, volume_btc_list, volume_eth_list, volume_bnb_list, volume_not_trade_list, volume_start, fee_usdt, trade_count

################################################################################################################


def trade_slip(test, predict, FEE, SLIP_N):
    volume_list = list()
    volume_btc_list = list()
    volume_eth_list = list()
    volume_bnb_list = list()
    volume_not_trade_list = list()

    part = 0.5  #доля для продажи

    volume_btc = 100
    volume_eth = 100
    volume_bnb = 100
    volume_start = volume_btc + volume_eth + volume_bnb
    fee_usdt = 0
    trade_count = 0

    # Для определения профита с покупки (price_sell - price_buy)/ price_sell - profit_fee_coef * fee)
    # нужно сохранять цену последней сделки по каждой паре
    bnb_btc_last_trade = test[0,3]
    eth_btc_last_trade = test[0, 4]
    bnb_eth_last_trade = test[0, 5]
    profit_fee_coef = 30

    for i, pred in enumerate(predict):

        if i!=0 and i != len(predict)-1:
            if pred == 1 and profit_status(bnb_btc_last_trade, test[i,3], FEE, profit_fee_coef):# and last_trade != 'btc_bnb':   # продажа btc, покупка bnb
                volume_btc, volume_bnb, fee, status = sell_buy(volume_btc, volume_bnb, 1/test[i+1,3], test[i+1,0], test[i+1,2], part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_eth = volume_eth * (test[i+1, 1]/test[i, 1])

            elif pred == 2 and volume_bnb/(volume_btc + volume_eth + volume_bnb) > 0.1  and profit_status(test[i,3], bnb_btc_last_trade, FEE, profit_fee_coef):# and last_trade != 'bnb_btc':   # продажа bnb, покупка btc
                volume_bnb, volume_btc, fee, status = sell_buy(volume_bnb, volume_btc, test[i+1, 3], test[i+1, 2], test[i+1, 0], part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_eth = volume_eth * (test[i+1, 1] / test[i, 1])

            elif pred == 3 and profit_status(eth_btc_last_trade, test[i,4], FEE, profit_fee_coef):# and last_trade != 'btc_eth':   # продажа btc, покупка eth
                volume_btc, volume_eth, fee, status = sell_buy(volume_btc, volume_eth, 1/test[i+1, 4], test[i+1, 0], test[i+1, 1], part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_bnb = volume_bnb * (test[i+1, 2] / test[i, 2])

            elif pred == 4  and profit_status(test[i,4], eth_btc_last_trade, FEE, profit_fee_coef):# and last_trade != 'eth_btc':   # продажа eth, покупка btc
                volume_eth, volume_btc, fee, status = sell_buy(volume_eth, volume_btc, test[i+1, 4], test[i+1, 1], test[i+1, 0], part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_bnb = volume_bnb * (test[i+1, 2] / test[i, 2])

            elif pred == 5 and profit_status(bnb_eth_last_trade, test[i,5], FEE, profit_fee_coef):# and last_trade != 'eth_bnb':   # продажа eth, покупка bnb
                volume_eth, volume_bnb, fee, status = sell_buy(volume_eth, volume_bnb, 1/test[i+1, 5], test[i+1, 1], test[i+1, 2], part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_btc = volume_btc * (test[i+1, 0] / test[i, 0])

            elif pred == 6 and volume_bnb/(volume_btc + volume_eth + volume_bnb) > 0.1 and profit_status(test[i,5], bnb_eth_last_trade, FEE, profit_fee_coef):# and last_trade != 'bnb_eth':   # продажа bnb, покупка eth
                volume_bnb, volume_eth, fee, status = sell_buy(volume_bnb, volume_eth, test[i+1, 5], test[i+1, 2], test[i+1, 1], part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_btc = volume_btc * (test[i+1, 0] / test[i, 0])

            else:
                fee = 0
                volume_btc = volume_btc * (test[i, 0] / test[i - 1, 0])
                volume_eth = volume_eth * (test[i, 1] / test[i - 1, 1])
                volume_bnb = volume_bnb * (test[i, 2] / test[i - 1, 2])
        else:
            fee = 0
        fee_usdt += fee

        volume_not_trade = (100 * (1 + (test[i, 0] - test[0, 0]) / test[0, 0]) +
                            100 * (1 + (test[i, 1] - test[0, 1]) / test[0, 1]) +
                            100 * (1 + (test[i, 2] - test[0, 2]) / test[0, 2])
                            )


        volume_list.append(volume_btc + volume_eth + volume_bnb)
        volume_btc_list.append(volume_btc)
        volume_eth_list.append(volume_eth)
        volume_bnb_list.append(volume_bnb)
        volume_not_trade_list.append(volume_not_trade)

    return volume_list, volume_btc_list, volume_eth_list, volume_bnb_list, volume_not_trade_list, volume_start, fee_usdt, trade_count


###############################################################################################################################

def trade_v2(test, predict, FEE, SLIP_N):
    volume_list = list()
    volume_btc_list = list()
    volume_eth_list = list()
    volume_bnb_list = list()
    volume_not_trade_list = list()

    part = 0.2  #доля для продажи

    volume_btc = 100
    volume_eth = 100
    volume_bnb = 100
    volume_start = volume_btc + volume_eth + volume_bnb
    fee_usdt = 0
    trade_count = 0

    # Для определения профита с покупки (price_sell - price_buy)/ price_sell - profit_fee_coef * fee)
    # нужно сохранять цену последней сделки по каждой паре
    bnb_btc_last_trade = test[0,3]
    eth_btc_last_trade = test[0, 4]
    bnb_eth_last_trade = test[0, 5]
    profit_fee_coef = 10

    # Число последовательных сделок в одном предсказании
    bnb_btc_min_trade_count = 1
    bnb_btc_max_trade_count = 1
    eth_btc_min_trade_count = 1
    eth_btc_max_trade_count = 1
    bnb_eth_min_trade_count = 1
    bnb_eth_max_trade_count = 1

    for i, pred in enumerate(predict):

        if i!=0:
            if pred == 1 and profit_status(bnb_btc_last_trade, test[i,3], FEE, profit_fee_coef):# and last_trade != 'btc_bnb':   # продажа btc, покупка bnb
                volume_btc, volume_bnb, fee, status = sell_buy(volume_btc, volume_bnb, 1/test[i,3], test[i,0], test[i,2], bnb_btc_min_trade_count*part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_eth = volume_eth * (test[i,1]/test[i-1,1])
                bnb_btc_last_trade = test[i,3]

                bnb_btc_min_trade_count += 1
                bnb_btc_max_trade_count = 1
                eth_btc_min_trade_count = 1
                eth_btc_max_trade_count = 1
                bnb_eth_min_trade_count = 1
                bnb_eth_max_trade_count = 1

            elif pred == 2 and volume_bnb/(volume_btc + volume_eth + volume_bnb) > 0.1  and profit_status(test[i,3], bnb_btc_last_trade, FEE, profit_fee_coef):# and last_trade != 'bnb_btc':   # продажа bnb, покупка btc
                volume_bnb, volume_btc, fee, status = sell_buy(volume_bnb, volume_btc, test[i, 3], test[i, 2], test[i, 0], bnb_btc_max_trade_count*part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_eth = volume_eth * (test[i, 1] / test[i - 1, 1])
                bnb_btc_last_trade = test[i, 3]

                bnb_btc_min_trade_count = 1
                bnb_btc_max_trade_count += 1
                eth_btc_min_trade_count = 1
                eth_btc_max_trade_count = 1
                bnb_eth_min_trade_count = 1
                bnb_eth_max_trade_count = 1

            elif pred == 3 and profit_status(eth_btc_last_trade, test[i,4], FEE, profit_fee_coef):# and last_trade != 'btc_eth':   # продажа btc, покупка eth
                volume_btc, volume_eth, fee, status = sell_buy(volume_btc, volume_eth, 1/test[i, 4], test[i, 0], test[i, 1], eth_btc_min_trade_count*part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_bnb = volume_bnb * (test[i, 2] / test[i - 1, 2])
                eth_btc_last_trade = test[i, 4]

                bnb_btc_min_trade_count = 1
                bnb_btc_max_trade_count = 1
                eth_btc_min_trade_count += 1
                eth_btc_max_trade_count = 1
                bnb_eth_min_trade_count = 1
                bnb_eth_max_trade_count = 1

            elif pred == 4  and profit_status(test[i,4], eth_btc_last_trade, FEE, profit_fee_coef):# and last_trade != 'eth_btc':   # продажа eth, покупка btc
                volume_eth, volume_btc, fee, status = sell_buy(volume_eth, volume_btc, test[i, 4], test[i, 1], test[i, 0], eth_btc_max_trade_count*part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_bnb = volume_bnb * (test[i, 2] / test[i - 1, 2])
                eth_btc_last_trade = test[i, 4]

                bnb_btc_min_trade_count = 1
                bnb_btc_max_trade_count = 1
                eth_btc_min_trade_count = 1
                eth_btc_max_trade_count += 1
                bnb_eth_min_trade_count = 1
                bnb_eth_max_trade_count = 1

            elif pred == 5 and profit_status(bnb_eth_last_trade, test[i,5], FEE, profit_fee_coef):# and last_trade != 'eth_bnb':   # продажа eth, покупка bnb
                volume_eth, volume_bnb, fee, status = sell_buy(volume_eth, volume_bnb, 1/test[i, 5], test[i, 1], test[i, 2], bnb_eth_min_trade_count*part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_btc = volume_btc * (test[i, 0] / test[i - 1, 0])
                bnb_eth_last_trade = test[i, 5]

                bnb_btc_min_trade_count = 1
                bnb_btc_max_trade_count = 1
                eth_btc_min_trade_count = 1
                eth_btc_max_trade_count = 1
                bnb_eth_min_trade_count += 1
                bnb_eth_max_trade_count = 1

            elif pred == 6 and volume_bnb/(volume_btc + volume_eth + volume_bnb) > 0.1 and profit_status(test[i,5], bnb_eth_last_trade, FEE, profit_fee_coef):# and last_trade != 'bnb_eth':   # продажа bnb, покупка eth
                volume_bnb, volume_eth, fee, status = sell_buy(volume_bnb, volume_eth, test[i, 5], test[i, 2], test[i, 1], bnb_eth_max_trade_count*part, FEE,
                                                  SLIP_N)
                trade_count += status
                volume_btc = volume_btc * (test[i, 0] / test[i - 1, 0])
                bnb_eth_last_trade = test[i, 5]

                bnb_btc_min_trade_count = 1
                bnb_btc_max_trade_count = 1
                eth_btc_min_trade_count = 1
                eth_btc_max_trade_count = 1
                bnb_eth_min_trade_count = 1
                bnb_eth_max_trade_count += 1

            else:
                fee = 0
                volume_btc = volume_btc * (test[i, 0] / test[i - 1, 0])
                volume_eth = volume_eth * (test[i, 1] / test[i - 1, 1])
                volume_bnb = volume_bnb * (test[i, 2] / test[i - 1, 2])
        else:
            fee = 0
        fee_usdt += fee
        volume_not_trade = (100 * (1 + (test[i, 0] - test[0, 0]) / test[0, 0]) +
                            100 * (1 + (test[i, 1] - test[0, 1]) / test[0, 1]) +
                            100 * (1 + (test[i, 2] - test[0, 2]) / test[0, 2])
                            )

        volume_list.append(volume_btc + volume_eth + volume_bnb)
        volume_btc_list.append(volume_btc)
        volume_eth_list.append(volume_eth)
        volume_bnb_list.append(volume_bnb)
        volume_not_trade_list.append(volume_not_trade)

    return volume_list, volume_btc_list, volume_eth_list, volume_bnb_list, volume_not_trade_list, volume_start, fee_usdt, trade_count

###############################################################################################################################

def trade_btc_usdt(test, predict, FEE, part, profit_fee_coef, trade_move_coef):  #  trade_move_coef - для борьбы с просадками, величина ограничавает ранний выкуп после просадки.

    volume_list = list()
    volume_btc_list = list()
    volume_usdt_list = list()
    volume_not_trade_list = list()



    volume_btc_start = 650
    volume_usdt_start = 650
    volume_btc = volume_btc_start
    volume_usdt = volume_usdt_start

    volume_start = volume_btc_start + volume_usdt_start
    trading_on = 1
    fee_usdt = 0
    trade_count = 0
    fee = 0  # комиссия в usdt со сделки, FEE = 0 ,00075 комиссия для расчёта прибылоьности сделки и проскальзывания

    # Для определения профита с покупки (price_sell - price_buy)/ price_sell - profit_fee_coef * fee)
    # нужно сохранять цену последней сделки по каждой паре
    btc_usdt_last_trade = test[0,0]
    trade_min_first = test[0,0]
    trade_max_first = test[0,0]

    part = part
    # part = 0.22  #доля для продажи
    profit_fee_coef = profit_fee_coef
    # profit_fee_coef = 1 # 1.5 в работе на торговле

    # Число последовательных сделок в одном предсказании
    btc_usdt_min_trade_count = 0
    btc_usdt_max_trade_count = 0

    for i, pred in enumerate(predict):

        if i!=0 and i != len(predict)-1:
            if pred == 1 and profit_status(btc_usdt_last_trade, test[i,0], FEE, profit_fee_coef,btc_usdt_min_trade_count): # продажа usdt, покупка btc
                volume_usdt_temp, volume_btc_temp, fee, status = sell_buy_not_fee(volume_usdt, volume_btc, 1/test[i+1,0], test[i+1,0]/test[i+1,0], test[i+1,0], (btc_usdt_min_trade_count+1)*part, FEE)
                if (test[i,0] - trade_max_first)/test[i,0] > fee*trade_move_coef:    # защита от выкупа по невыгодной цене при скочко образном изменении цены
                    trading_on = 0
                if status == 1 and trading_on == 1:
                    if btc_usdt_min_trade_count == 0:
                        trade_min_first = test[i,0]
                    trade_count += status
                    btc_usdt_last_trade = test[i,0]
                    #btc_usdt_min_trade_count += 1
                    btc_usdt_min_trade_count = 0
                    btc_usdt_max_trade_count = 0
                    volume_btc = volume_btc_temp
                    volume_usdt = volume_usdt_temp
                else:
                    fee = 0
                    volume_btc = volume_btc * (test[i, 0] / test[i - 1, 0])
                    volume_usdt = volume_usdt

            elif pred == 2 and profit_status(test[i,0], btc_usdt_last_trade, FEE, profit_fee_coef, btc_usdt_max_trade_count): # продажа btc, покупка bnb
                volume_btc_temp, volume_usdt_temp, fee, status = sell_buy_not_fee(volume_btc, volume_usdt, test[i+1, 0], test[i+1, 0], test[i+1,0]/test[i+1, 0], (btc_usdt_max_trade_count+1)*part, FEE)
                if (trade_min_first - test[i,0])/test[i,0] > fee*trade_move_coef:   # защита от выкупа по невыгодной цене при скочко образном изменении цены
                    trading_on = 0
                if status == 1 and trading_on == 1:
                    if btc_usdt_max_trade_count == 0:
                        trade_max_first = test[i,0]
                    trade_count += status
                    btc_usdt_last_trade = test[i, 0]
                    btc_usdt_min_trade_count = 0
                    #btc_usdt_max_trade_count += 1
                    btc_usdt_max_trade_count = 0
                    volume_btc = volume_btc_temp
                    volume_usdt = volume_usdt_temp
                else:
                    fee = 0
                    volume_btc = volume_btc * (test[i, 0] / test[i - 1, 0])
                    volume_usdt = volume_usdt

            else:
                fee = 0
                volume_btc = volume_btc * (test[i, 0] / test[i-1, 0])
                volume_usdt = volume_usdt

        fee_usdt += fee
        volume_not_trade = (volume_btc_start * (1 + (test[i, 0] - test[0, 0]) / test[0, 0]) +
                            volume_usdt_start
                            )



        volume_list.append(volume_btc + volume_usdt)
        volume_btc_list.append(volume_btc)
        volume_usdt_list.append(volume_usdt)
        volume_not_trade_list.append(volume_not_trade)

    return volume_list, volume_btc_list, volume_usdt_list, volume_not_trade_list, volume_start, fee_usdt, trade_count

#########################################################################################################################

# V_2 объём продажи определяется как: остаток домноженный на (1.12**(btc_usdt_min_trade_count+1))*part или (1.12**(btc_usdt_max_trade_count+1))*part

def trade_btc_usdt_v2(test, predict, FEE, part, profit_fee_coef, trade_move_coef):  #  trade_move_coef - для борьбы с просадками, величина ограничавает ранний выкуп после просадки.

    volume_list = list()
    volume_btc_list = list()
    volume_usdt_list = list()
    volume_not_trade_list = list()



    volume_btc_start = 600
    volume_usdt_start = 600
    volume_btc = volume_btc_start
    volume_usdt = volume_usdt_start

    volume_start = volume_btc_start + volume_usdt_start
    trading_on = 1
    fee_usdt = 0
    trade_count = 0
    fee = 0  # комиссия в usdt со сделки, FEE = 0 ,00075 комиссия для расчёта прибылоьности сделки и проскальзывания

    # Для определения профита с покупки (price_sell - price_buy)/ price_sell - profit_fee_coef * fee)
    # нужно сохранять цену последней сделки по каждой паре
    btc_usdt_last_trade = test[0,0]
    trade_min_first = test[0,0]
    trade_max_first = test[0,0]

    part = part
    # part = 0.22  #доля для продажи
    profit_fee_coef = profit_fee_coef
    # profit_fee_coef = 1 # 1.5 в работе на торговле

    # Число последовательных сделок в одном предсказании
    btc_usdt_min_trade_count = 0
    btc_usdt_max_trade_count = 0

    for i, pred in enumerate(predict):

        if i!=0 and i != len(predict)-1:
            if pred == 1 and profit_status(btc_usdt_last_trade, test[i,0], FEE, profit_fee_coef, 0): #btc_usdt_min_trade_count): # продажа usdt, покупка btc
                volume_usdt_temp, volume_btc_temp, fee, status = sell_buy_not_fee(volume_usdt, volume_btc, 1/test[i+1,0], test[i+1,0]/test[i+1,0], test[i+1,0], (1.12**(btc_usdt_min_trade_count+1))*part, FEE)
                if (test[i,0] - trade_max_first)/test[i,0] > fee*trade_move_coef:    # защита от выкупа по невыгодной цене при скочко образном изменении цены
                    trading_on = 0
                if status == 1 and trading_on == 1:
                    if btc_usdt_min_trade_count == 0:
                        trade_min_first = test[i,0]
                    trade_count += status
                    btc_usdt_last_trade = test[i,0]
                    btc_usdt_min_trade_count += 1
                    #btc_usdt_min_trade_count = 0
                    btc_usdt_max_trade_count = 0
                    volume_btc = volume_btc_temp
                    volume_usdt = volume_usdt_temp
                else:
                    fee = 0
                    volume_btc = volume_btc * (test[i, 0] / test[i - 1, 0])
                    volume_usdt = volume_usdt

            elif pred == 2 and profit_status(test[i,0], btc_usdt_last_trade, FEE, profit_fee_coef, 0): #btc_usdt_max_trade_count): # продажа btc, покупка bnb
                volume_btc_temp, volume_usdt_temp, fee, status = sell_buy_not_fee(volume_btc, volume_usdt, test[i+1, 0], test[i+1, 0], test[i+1,0]/test[i+1, 0], (1.12**(btc_usdt_max_trade_count+1))*part, FEE)
                if (trade_min_first - test[i,0])/test[i,0] > fee*trade_move_coef:   # защита от выкупа по невыгодной цене при скочко образном изменении цены
                    trading_on = 0
                if status == 1 and trading_on == 1:
                    if btc_usdt_max_trade_count == 0:
                        trade_max_first = test[i,0]
                    trade_count += status
                    btc_usdt_last_trade = test[i, 0]
                    btc_usdt_min_trade_count = 0
                    btc_usdt_max_trade_count += 1
                    #btc_usdt_max_trade_count = 0
                    volume_btc = volume_btc_temp
                    volume_usdt = volume_usdt_temp
                else:
                    fee = 0
                    volume_btc = volume_btc * (test[i, 0] / test[i - 1, 0])
                    volume_usdt = volume_usdt

            else:
                fee = 0
                volume_btc = volume_btc * (test[i, 0] / test[i-1, 0])
                volume_usdt = volume_usdt

        fee_usdt += fee
        volume_not_trade = (volume_btc_start * (1 + (test[i, 0] - test[0, 0]) / test[0, 0]) +
                            volume_usdt_start
                            )



        volume_list.append(volume_btc + volume_usdt)
        volume_btc_list.append(volume_btc)
        volume_usdt_list.append(volume_usdt)
        volume_not_trade_list.append(volume_not_trade)

    return volume_list, volume_btc_list, volume_usdt_list, volume_not_trade_list, volume_start, fee_usdt, trade_count


#########################################################################################################################

# V_3 объём продажи определяется как: остаток домноженный на (1.12**(btc_usdt_min_trade_count+1))*part*coef или (1.12**(btc_usdt_max_trade_count+1))*part*coef
#  где coef = 1-(volume_usdt - (volume_btc+volume_usdt)/2)/((volume_btc+volume_usdt)*3)
#      coef = 1-(volume_btc - (volume_btc+volume_usdt)/2)/((volume_btc+volume_usdt)*3)
def trade_btc_usdt_v3(test, predict, FEE, part, profit_fee_coef, trade_move_coef):  #  trade_move_coef - для борьбы с просадками, величина ограничавает ранний выкуп после просадки.

    volume_list = list()
    volume_btc_list = list()
    volume_usdt_list = list()
    volume_not_trade_list = list()



    volume_btc_start = 600
    volume_usdt_start = 600
    volume_btc = volume_btc_start
    volume_usdt = volume_usdt_start

    volume_start = volume_btc_start + volume_usdt_start
    trading_on = 1
    fee_usdt = 0
    trade_count = 0
    fee = 0  # комиссия в usdt со сделки, FEE = 0 ,00075 комиссия для расчёта прибылоьности сделки и проскальзывания

    # Для определения профита с покупки (price_sell - price_buy)/ price_sell - profit_fee_coef * fee)
    # нужно сохранять цену последней сделки по каждой паре
    btc_usdt_last_trade = test[0,0]
    trade_min_first = test[0,0]
    trade_max_first = test[0,0]

    part = part
    # part = 0.22  #доля для продажи
    profit_fee_coef = profit_fee_coef
    # profit_fee_coef = 1 # 1.5 в работе на торговле

    # Число последовательных сделок в одном предсказании
    btc_usdt_min_trade_count = 0
    btc_usdt_max_trade_count = 0

    for i, pred in enumerate(predict):

        if i!=0 and i != len(predict)-1:
            if pred == 1 and profit_status(btc_usdt_last_trade, test[i,0], FEE, profit_fee_coef, 0): #btc_usdt_min_trade_count): # продажа usdt, покупка btc
                if (volume_usdt - (volume_btc+volume_usdt)/2) > 0:
                    coef = 1-(volume_usdt - (volume_btc+volume_usdt)/2)/((volume_btc+volume_usdt)*3)
                else:
                    coef = 1
                volume_usdt_temp, volume_btc_temp, fee, status = sell_buy_not_fee(volume_usdt, volume_btc, 1/test[i,0], test[i,0]/test[i,0], test[i,0], coef*(1.12**(btc_usdt_min_trade_count+1))*part, FEE)
                if (test[i,0] - trade_max_first)/test[i,0] > fee*trade_move_coef:    # защита от выкупа по невыгодной цене при скочко образном изменении цены
                    trading_on = 0
                if status == 1 and trading_on == 1:
                    if btc_usdt_min_trade_count == 0:
                        trade_min_first = test[i,0]
                    trade_count += status
                    btc_usdt_last_trade = test[i,0]
                    btc_usdt_min_trade_count += 1
                    #btc_usdt_min_trade_count = 0
                    btc_usdt_max_trade_count = 0
                    volume_btc = volume_btc_temp
                    volume_usdt = volume_usdt_temp
                else:
                    fee = 0
                    volume_btc = volume_btc * (test[i, 0] / test[i - 1, 0])
                    volume_usdt = volume_usdt

            elif pred == 2 and profit_status(test[i,0], btc_usdt_last_trade, FEE, profit_fee_coef, 0): #btc_usdt_max_trade_count): # продажа btc, покупка bnb
                if (volume_btc - (volume_btc+volume_usdt)/2) > 0:
                    coef = 1-(volume_btc - (volume_btc+volume_usdt)/2)/((volume_btc+volume_usdt)*3)
                else:
                    coef = 1
                volume_btc_temp, volume_usdt_temp, fee, status = sell_buy_not_fee(volume_btc, volume_usdt, test[i, 0], test[i, 0], test[i,0]/test[i, 0], coef*(1.12**(btc_usdt_max_trade_count+1))*part, FEE)
                if (trade_min_first - test[i,0])/test[i,0] > fee*trade_move_coef:   # защита от выкупа по невыгодной цене при скочко образном изменении цены
                    trading_on = 0
                if status == 1 and trading_on == 1:
                    if btc_usdt_max_trade_count == 0:
                        trade_max_first = test[i,0]
                    trade_count += status
                    btc_usdt_last_trade = test[i, 0]
                    btc_usdt_min_trade_count = 0
                    btc_usdt_max_trade_count += 1
                    #btc_usdt_max_trade_count = 0
                    volume_btc = volume_btc_temp
                    volume_usdt = volume_usdt_temp
                else:
                    fee = 0
                    volume_btc = volume_btc * (test[i, 0] / test[i - 1, 0])
                    volume_usdt = volume_usdt

            else:
                fee = 0
                volume_btc = volume_btc * (test[i, 0] / test[i-1, 0])
                volume_usdt = volume_usdt

        fee_usdt += fee
        volume_not_trade = (volume_btc_start * (1 + (test[i, 0] - test[0, 0]) / test[0, 0]) +
                            volume_usdt_start
                            )



        volume_list.append(volume_btc + volume_usdt)
        volume_btc_list.append(volume_btc)
        volume_usdt_list.append(volume_usdt)
        volume_not_trade_list.append(volume_not_trade)

    return volume_list, volume_btc_list, volume_usdt_list, volume_not_trade_list, volume_start, fee_usdt, trade_count


#########################################################################################################################


def sell_buy_not_fee_v4(volume_x, volume_y, x_y_price, x_usdt_price, y_usdt_price, part, FEE, centralization):  # продаваемая валюта X, покупаемая валюта Y, ....

    min_order_status = 0
    volume = volume_x + volume_y

    # два случая: volume_x > volume/2, то нужно продавать больше, иначе нужно продавать меньше, так среднее будет тяготеть к центу
    if volume_x > volume/2:
        volume_x_sell = volume * part * (1+centralization)  # продаваемый объём больше
    else:
        volume_x_sell = volume * part * (1-centralization)  # продаваемый объём меньше
        if volume_x_sell > volume_x:  #  если под продаваемый объём недостаточно средств, то не продавать
            volume_x_sell = 0

    volume_y_buy = volume_x_sell * y_usdt_price * x_y_price / x_usdt_price  # покупаемый объём
    fee = volume_y_buy * FEE  # размер комиссии

    if volume_x_sell > 11:  # статус превышения минимально допустимого ордера Binance 10 usdt
        min_order_status = 1
    else:
        print('min_order_status - ', min_order_status)

    if min_order_status == 1:  # выполнить сделку, если статус 1
        volume_x = volume_x - volume_x_sell  # объём продаваемой валюты после сделки
        volume_y = volume_y + volume_y_buy  # объём покупаемой валюты после сделки
        status = 1
    else:
        status = 0
        fee = 0
    return volume_x, volume_y, fee, status

def trade_btc_usdt_v4(test, predict, FEE, part, profit_fee_coef, trade_move_coef, centralization):  #  trade_move_coef - для борьбы с просадками, величина ограничавает ранний выкуп после просадки.

    volume_list = list()
    volume_btc_list = list()
    volume_usdt_list = list()
    volume_not_trade_list = list()

    volume_btc_start = 600
    volume_usdt_start = 600
    volume_btc = volume_btc_start
    volume_usdt = volume_usdt_start

    volume_start = volume_btc_start + volume_usdt_start
    trading_on = 1
    fee_usdt = 0
    trade_count = 0
    fee = 0  # комиссия в usdt со сделки, FEE = 0 ,00075 комиссия для расчёта прибылоьности сделки и проскальзывания

    # Для определения профита с покупки (price_sell - price_buy)/ price_sell - profit_fee_coef * fee)
    # нужно сохранять цену последней сделки по каждой паре
    btc_usdt_last_trade = test[0,0]
    trade_min_first = test[0,0]
    trade_max_first = test[0,0]

    part = part
    # part = 0.22  #доля для продажи
    profit_fee_coef = profit_fee_coef
    # profit_fee_coef = 1 # 1.5 в работе на торговле

    # Число последовательных сделок в одном предсказании
    btc_usdt_min_trade_count = 0
    btc_usdt_max_trade_count = 0

    for i, pred in enumerate(predict):

        if i!=0 and i != len(predict)-1:
            if pred == 1 and profit_status(btc_usdt_last_trade, test[i,0], FEE, profit_fee_coef, 0): #btc_usdt_min_trade_count): # продажа usdt, покупка btc
                volume_usdt_temp, volume_btc_temp, fee, status = sell_buy_not_fee_v4(volume_usdt, volume_btc, 1/test[i,0], test[i,0]/test[i,0], test[i,0], part, FEE, centralization)
                if (test[i,0] - trade_max_first)/test[i,0] > fee*trade_move_coef:    # защита от выкупа по невыгодной цене при скочко образном изменении цены
                    trading_on = 0
                if status == 1 and trading_on == 1:   # status
                    if btc_usdt_min_trade_count == 0:
                        trade_min_first = test[i,0]
                    trade_count += status
                    btc_usdt_last_trade = test[i,0]
                    btc_usdt_min_trade_count += 1
                    #btc_usdt_min_trade_count = 0
                    btc_usdt_max_trade_count = 0
                    volume_btc = volume_btc_temp
                    volume_usdt = volume_usdt_temp
                else:
                    fee = 0
                    volume_btc = volume_btc * (test[i, 0] / test[i - 1, 0])
                    volume_usdt = volume_usdt

            elif pred == 2 and profit_status(test[i,0], btc_usdt_last_trade, FEE, profit_fee_coef, 0): #btc_usdt_max_trade_count): # продажа btc, покупка bnb
                volume_btc_temp, volume_usdt_temp, fee, status = sell_buy_not_fee_v4(volume_btc, volume_usdt, test[i, 0], test[i, 0], test[i,0]/test[i, 0], part, FEE, centralization)
                if (trade_min_first - test[i,0])/test[i,0] > fee*trade_move_coef:   # защита от выкупа по невыгодной цене при скочко образном изменении цены
                    trading_on = 0
                if status == 1 and trading_on == 1:
                    if btc_usdt_max_trade_count == 0:
                        trade_max_first = test[i,0]
                    trade_count += status
                    btc_usdt_last_trade = test[i, 0]
                    btc_usdt_min_trade_count = 0
                    btc_usdt_max_trade_count += 1
                    #btc_usdt_max_trade_count = 0
                    volume_btc = volume_btc_temp
                    volume_usdt = volume_usdt_temp
                else:
                    fee = 0
                    volume_btc = volume_btc * (test[i, 0] / test[i - 1, 0])
                    volume_usdt = volume_usdt

            else:
                fee = 0
                volume_btc = volume_btc * (test[i, 0] / test[i-1, 0])
                volume_usdt = volume_usdt

        fee_usdt += fee
        volume_not_trade = (volume_btc_start * (1 + (test[i, 0] - test[0, 0]) / test[0, 0]) +
                            volume_usdt_start
                            )



        volume_list.append(volume_btc + volume_usdt)
        volume_btc_list.append(volume_btc)
        volume_usdt_list.append(volume_usdt)
        volume_not_trade_list.append(volume_not_trade)

    return volume_list, volume_btc_list, volume_usdt_list, volume_not_trade_list, volume_start, fee_usdt, trade_count



#########################################################################################################################


def sell_buy_not_fee_v5(volume_x, volume_y, x_y_price, x_usdt_price, y_usdt_price, part, FEE, centralization):  # продаваемая валюта X, покупаемая валюта Y, ....

    min_order_status = 0
    volume_x_usdt = volume_x * x_usdt_price
    volume_y_usdt = volume_y * y_usdt_price
    volume_all_usdt = volume_x_usdt + volume_y_usdt

    # два случая: volume_x > volume/2, то нужно продавать больше, иначе нужно продавать меньше, так среднее будет тяготеть к центу
    if volume_x_usdt > volume_all_usdt/2:
        volume_x_usdt_sell = volume_all_usdt * part * (1+centralization)  # продаваемый объём больше
    else:
        volume_x_usdt_sell = volume_all_usdt * part * (1-centralization)  # продаваемый объём меньше
        if volume_x_usdt_sell > volume_x_usdt:  #  если под продаваемый объём недостаточно средств, то не продавать
            volume_x_usdt_sell = 0

    volume_y_usdt_buy = volume_x_usdt_sell * y_usdt_price * x_y_price / x_usdt_price  # покупаемый объём
    fee = volume_y_usdt_buy * FEE  # размер комиссии

    if volume_x_usdt_sell > 11:  # статус превышения минимально допустимого ордера Binance 10 usdt
        min_order_status = 1
    # else:
    #     print('min_order_status - ', min_order_status)
    #     print('volume_x_usdt_sell - ', volume_x_usdt_sell)

    if min_order_status == 1:  # выполнить сделку, если статус 1
        volume_x = volume_x - volume_x_usdt_sell / x_usdt_price  # объём продаваемой валюты после сделки
        volume_y = volume_y + volume_y_usdt_buy / y_usdt_price  # объём покупаемой валюты после сделки
        status = 1
    else:
        status = 0
        fee = 0
    return volume_x, volume_y, fee, status

def trade_btc_usdt_v5(test, predict, FEE, part, profit_fee_coef, trade_move_coef, centralization, std_len, std_limit, std_delay):  #  trade_move_coef - для борьбы с просадками, величина ограничавает ранний выкуп после просадки.

    volume_list = list()
    volume_not_trade_list = list()
    volume_btc_list = list()
    volume_usdt_list = list()
    std_result_list = list()

    # инициализация начальных объёмов в долларах
    volume_btc_usdt_start = 600
    volume_usdt_start = 600

    volume_btc = volume_btc_usdt_start/test[0,0]
    volume_usdt = volume_usdt_start
    volume_start = volume_btc_usdt_start + volume_usdt_start
    trading_on = 1
    fee_usdt = 0
    trade_count = 0
    std_count = 0   # число оставшихся шагов в блокировке по std
    fee = 0  # комиссия в usdt со сделки, FEE = 0 ,00075 комиссия для расчёта прибылоьности сделки и проскальзывания
    std_result = False

    # Для определения профита с покупки (price_sell - price_buy)/ price_sell - profit_fee_coef * fee)
    # нужно сохранять цену последней сделки по каждой паре
    btc_usdt_last_trade = test[0,0]
    trade_min_first = test[0,0]
    trade_max_first = test[0,0]

    # Число последовательных сделок в одном предсказании
    btc_usdt_min_trade_count = 0
    btc_usdt_max_trade_count = 0

    for i, pred in enumerate(predict):


        if i > std_len-2 and i != len(predict)-1:  # обрезка начала под удовлетворение применения среза под среднеквадратичное отклонение

            # получение среднеквадратичного отклонения для последних std_len тиков, включая текущий и блокировка в
            if std_count == 0: # если нет блокировки по std

                #std_result = test[i - std_len + 1:i + 1, 0].std() > std_limit
                std_result = abs(test[i, 0] - test[i - 1, 0])/test[i, 0] > std_limit*FEE
                if std_result: # если возникла блокировка обновить счётчик std_count
                    std_count = std_delay
            else:  # уменьшить счётчик на один шаг
                std_count -= 1

            if std_count == 0 and pred == 1 and profit_status(btc_usdt_last_trade, test[i,0], FEE, profit_fee_coef, 0): #btc_usdt_min_trade_count): # продажа usdt, покупка btc

                volume_usdt_temp, volume_btc_temp, fee, status = sell_buy_not_fee_v5(volume_usdt, volume_btc, 1/test[i,0], test[i,0]/test[i,0], test[i,0], part, FEE, centralization)
                if (test[i,0] - trade_max_first)/test[i,0] > fee*trade_move_coef:    # защита от выкупа по невыгодной цене при скочко образном изменении цены
                    trading_on = 0
                if status == 1 and trading_on == 1:   # status
                    if btc_usdt_min_trade_count == 0:
                        trade_min_first = test[i,0]
                    trade_count += status
                    btc_usdt_last_trade = test[i,0]
                    btc_usdt_min_trade_count += 1
                    #btc_usdt_min_trade_count = 0
                    btc_usdt_max_trade_count = 0
                    volume_btc = volume_btc_temp
                    volume_usdt = volume_usdt_temp
                else:
                    fee = 0
                    volume_btc = volume_btc
                    volume_usdt = volume_usdt

            elif std_count == 0 and pred == 2 and profit_status(test[i,0], btc_usdt_last_trade, FEE, profit_fee_coef, 0): #btc_usdt_max_trade_count): # продажа btc, покупка bnb
                volume_btc_temp, volume_usdt_temp, fee, status = sell_buy_not_fee_v5(volume_btc, volume_usdt, test[i, 0], test[i, 0], test[i,0]/test[i, 0], part, FEE, centralization)
                if (trade_min_first - test[i,0])/test[i,0] > fee*trade_move_coef:   # защита от выкупа по невыгодной цене при скочко образном изменении цены
                    trading_on = 0
                if status == 1 and trading_on == 1:
                    if btc_usdt_max_trade_count == 0:
                        trade_max_first = test[i,0]
                    trade_count += status
                    btc_usdt_last_trade = test[i, 0]
                    btc_usdt_min_trade_count = 0
                    btc_usdt_max_trade_count += 1
                    #btc_usdt_max_trade_count = 0
                    volume_btc = volume_btc_temp
                    volume_usdt = volume_usdt_temp
                else:
                    fee = 0
                    volume_btc = volume_btc
                    volume_usdt = volume_usdt

            else:
                fee = 0
                volume_btc = volume_btc
                volume_usdt = volume_usdt

        fee_usdt += fee
        # неуверен за корректность расчёта volume_not_trade
        volume_not_trade = (volume_btc_usdt_start * (1 + (test[i, 0] - test[0, 0]) / test[0, 0]) +
                            volume_usdt_start
                            )

        volume_list.append(volume_btc * test[i,0] + volume_usdt)
        volume_btc_list.append(volume_btc * test[i,0])
        volume_usdt_list.append(volume_usdt)
        volume_not_trade_list.append(volume_not_trade)
        std_result_list.append(std_result)

    return volume_list, volume_btc_list, volume_usdt_list, volume_not_trade_list, volume_start, fee_usdt, trade_count, std_result_list