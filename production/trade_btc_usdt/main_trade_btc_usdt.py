import numpy as np
import torch
import torch.nn as nn

from binance_trade import *
from sql_trade_write import *
from datetime import datetime
from time import time, sleep


class Bottleneck(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, inter_stride):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(inter_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, stride=inter_stride, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(inter_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv1d(inter_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Downsample, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNet1D(nn.Module):
    def __init__(self):
        super(ResNet1D, self).__init__()

        # input
        # self.conv1 = nn.Conv1d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv1d(6, 64, kernel_size=80, stride=2, padding=3, bias=False)  # kernel_size = 4 по умолчанию 7
        # self.bn1 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # layer 1
        self.downsample_1 = Downsample(64, 256, 1)
        self.bottleneck_1_0 = Bottleneck(64, 64, 256, 1)
        self.bottleneck_1_1 = Bottleneck(256, 64, 256, 1)
        self.bottleneck_1_2 = Bottleneck(256, 64, 256, 1)

        # layer 2
        self.downsample_2 = Downsample(256, 512, 2)
        self.bottleneck_2_0 = Bottleneck(256, 128, 512, 2)
        self.bottleneck_2_1 = Bottleneck(512, 128, 512, 1)
        self.bottleneck_2_2 = Bottleneck(512, 128, 512, 1)
        self.bottleneck_2_3 = Bottleneck(512, 128, 512, 1)

        # layer 3
        self.downsample_3 = Downsample(512, 1024, 2)
        self.bottleneck_3_0 = Bottleneck(512, 256, 1024, 2)
        self.bottleneck_3_1 = Bottleneck(1024, 256, 1024, 1)
        self.bottleneck_3_2 = Bottleneck(1024, 256, 1024, 1)
        self.bottleneck_3_3 = Bottleneck(1024, 256, 1024, 1)
        self.bottleneck_3_4 = Bottleneck(1024, 256, 1024, 1)
        self.bottleneck_3_5 = Bottleneck(1024, 256, 1024, 1)

        # layer 4
        self.downsample_4 = Downsample(1024, 2048, 2)
        self.bottleneck_4_0 = Bottleneck(1024, 512, 2048, 2)
        self.bottleneck_4_1 = Bottleneck(2048, 512, 2048, 1)
        self.bottleneck_4_2 = Bottleneck(2048, 512, 2048, 1)

        # linear
        # self.bn2 = nn.BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=2048, out_features=500, bias=True)
        # self.linear1 = nn.Linear(in_features=2048, out_features=7, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(512, 7, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x, x_min_max):
        # input
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        # layer 1
        x_downsampled = self.downsample_1(x)
        x = self.bottleneck_1_0(x)
        x = self.bottleneck_1_1(x)
        x = self.bottleneck_1_2(x)
        x = x + x_downsampled
        # layer 2
        x_downsampled = self.downsample_2(x)
        x = self.bottleneck_2_0(x)
        x = self.bottleneck_2_1(x)
        x = self.bottleneck_2_2(x)
        x = self.bottleneck_2_3(x)
        x = x + x_downsampled
        # layer 3
        x_downsampled = self.downsample_3(x)
        x = self.bottleneck_3_0(x)
        x = self.bottleneck_3_1(x)
        x = self.bottleneck_3_2(x)
        x = self.bottleneck_3_3(x)
        x = self.bottleneck_3_4(x)
        x = self.bottleneck_3_5(x)
        x = x + x_downsampled
        # layer 4
        x_downsampled = self.downsample_4(x)
        x = self.bottleneck_4_0(x)
        x = self.bottleneck_4_1(x)
        x = self.bottleneck_4_2(x)
        x = x + x_downsampled
        # linear
        # x_min_max = self.bn2(x_min_max)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = torch.cat((x, x_min_max), 1)
        x = self.relu2(x)
        # x = self.sigmoid(x)

        x = self.linear2(x)
        return x

    def inference(self, x, x_min_max):
        x = self.forward(x, x_min_max)
        x = self.sm(x)
        return x


# расчёт объёмов в паре при сделке и комиссии в usdt
def sell_buy(volume_x, x_y_price, y_usdt_price, part, FEE, SLIP_N):  # продаваемая валюта X, покупаемая валюта Y, ....
    volume_x_sell = volume_x * part  # продаваемый объём в крипте
    volume_y_buy = volume_x_sell * x_y_price  # покупаемый объём в крипте
    fee = volume_y_buy * FEE * SLIP_N  # размер комиссии в крипте
    return volume_x_sell, volume_y_buy, fee * y_usdt_price  # продаваемый объём в крипте, покупаемый объём в крипте, комиссия в usdt


# расчёт прибыли в usdt при сделке
def profit_calculation(coins, ticker_price, predict, part, fee, slip_n):
    volume_btc = float(coins[0]['free'])
    volume_usdt = float(coins[1]['free'])

    volume_btc_buy = 0
    volume_usdt_buy = 0
    volume_btc_sell = 0
    volume_usdt_sell = 0
    volume_before = volume_btc * ticker_price[0] + volume_usdt
    fee_usdt = 0
    volume_sell_crypt_after_usdt = 0 # остаток продаваемой крипты в usdt после сделки
    volume_status = 0  # статус не превышения минимально объёма крипты в кошельке
    min_order_status = 0  # статус превышения минимально допустимого размера ордера Binance
    if predict == 1:   # продажа usdt, покупка btc
        volume_usdt_sell, volume_btc_buy, fee_usdt = sell_buy(volume_usdt, 1 / ticker_price[0], ticker_price[0], part, fee, slip_n)
        volume_sell_crypt_after_usdt = (volume_usdt - volume_usdt_sell) # * ticker_price[0]

        if volume_usdt_sell > 11: # статус превышения минимально допустимого ордера Binance 10 usdt
            min_order_status = 1

    elif predict == 2:   # продажа btc, покупка usdt
        volume_btc_sell, volume_usdt_buy, fee_usdt = sell_buy(volume_btc, ticker_price[0], ticker_price[0]/ticker_price[0], part, fee, slip_n)
        volume_sell_crypt_after_usdt = (volume_btc - volume_btc_sell) * ticker_price[0]

        if volume_usdt_buy > 11:# статус превышения минимально допустимого ордера Binance 10 BTC
            min_order_status = 1

    if volume_sell_crypt_after_usdt > 1:  # статус не превышения минимально объёма крипты в кошельке в долларах
        volume_status = 1

    return volume_before, fee_usdt, min_order_status, volume_status, volume_usdt_buy, volume_btc_buy, volume_usdt_sell, volume_btc_sell   #прибыли и комиссия в usdt

# статус определения профита с покупки (price_sell - price_buy)/ price_sell - profit_fee_coef * fee)
def profit_status(price_max, price_min, fee, n, trade_count):
    status_buy = int((price_max - price_min)/price_max > n*fee*(trade_count+1))    # trade_count не опробовано
    return status_buy

# защита от выкупа по невыгодной цене после свечки
def trading_status(price_1, price_2, start, part, fee, profit_fee_coef, trade_move_coef):
    trading_on = int((price_1-price_2)/price_1 < fee*trade_move_coef)
    if trading_on == 0: # сохранение блокировки в БД
        sql_trading_settings_write(datetime.fromtimestamp(time()), trading_on, start, part, fee, profit_fee_coef, trade_move_coef)

    return trading_on

def main():
    print('Binance trade')

    random, image_length, features_length, batch_size, epoch, fee, slip_n = config_model()
    features_length = 80
    tickers_prices = list()
    res_net = torch.load('res_net', map_location=torch.device('cpu')).eval()
    # Число последовательных сделок в одном предсказании
    btc_usdt_min_trade_count = 0
    btc_usdt_max_trade_count = 0


    while True:

        time_begin = time()

        # чтение коэффициентов из БД
        #profit_fee_coef для условия совершения сделки, доля комиссии для превышения профита (учёт проскальзывания, 0 - отсутствует проскальзывание)

        trading_on, start, part, fee, profit_fee_coef, trade_move_coef = read_settings_sql()

        profit_fee_status = 0 # статус определения профита с покупки (price_sell - price_buy)/ price_sell - profit_fee_coef * fee)

        # запрос данных из binance
        coins, ticker_price = binance_read()
        tickers_prices.append(ticker_price)
        #print(coins)

        # подготовка features
        if(len(tickers_prices) >= features_length):

            # установка начальных данных для торгов
            if start == 1:
                res_net = torch.load('res_net', map_location=torch.device('cpu')).eval()
                btc_usdt_last_trade = ticker_price[0]
                trade_min_first = ticker_price[0]
                trade_max_first = ticker_price[0]
                start = 0
                sql_trading_settings_write(datetime.fromtimestamp(time()), trading_on, start, part, fee,
                                           profit_fee_coef, trade_move_coef)

            features = np.array(tickers_prices[-features_length:])
            features_min = features.T.min(axis=1)
            features_scaled = ((features - features_min) * 1000 / features_min)

            features_last_min = features_scaled[-1] - features_scaled.min()
            features_last_max = features_scaled.max() - features_scaled[-1]
            features_min_max = np.concatenate((features_last_min, features_last_max), axis=0)

            features_scaled = features_scaled.T

            # предсказание
            features_scaled_tensor = (torch.tensor(features_scaled.astype('float32')))[None,:,:]
            features_min_max_tensor = (torch.tensor(features_min_max.astype('float32')))[None,:]

            predict = int(torch.argmax(res_net.inference(features_scaled_tensor, features_min_max_tensor), dim=1).detach())

            # расчёт параметров сделки
            if predict == 1:  # btc_usdt_min_trade_count *  part
                volume_before, fee_usdt, min_order_status, volume_status, volume_usdt_buy, volume_btc_buy, volume_usdt_sell, volume_btc_sell = profit_calculation(
                    coins, ticker_price, predict, (1.12**(btc_usdt_min_trade_count+1))*part, fee,
                    slip_n)
            elif predict == 2:   # btc_usdt_max_trade_count * part
                volume_before, fee_usdt, min_order_status, volume_status, volume_usdt_buy, volume_btc_buy, volume_usdt_sell, volume_btc_sell = profit_calculation(
                    coins, ticker_price, predict, (1.12**(btc_usdt_max_trade_count+1))*part, fee,
                    slip_n)
            else:
                volume_before, fee_usdt, min_order_status, volume_status, volume_usdt_buy, volume_btc_buy, volume_usdt_sell, volume_btc_sell = profit_calculation(
                    coins, ticker_price, predict, (1.12**(btc_usdt_max_trade_count+1))*part * (1.12**(btc_usdt_min_trade_count+1)) * part, fee,
                    slip_n)

            # принятие решения по сделке и сделка
            if min_order_status + volume_status == 2: # сделка подходит по всем параметрам, все статусы 1
                order = {'symbol': 'None', 'orderId': 999, 'orderListId': 0, 'clientOrderId': '0', 'transactTime': 0, 'price': '0.00000000', 'origQty': '0.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'None', 'side': 'None', 'workingTime': 0, 'fills': [{'price': '0', 'qty': '0.000', 'commission': '0.00000', 'commissionAsset': 'None', 'tradeId': 0}], 'selfTradePreventionMode': 'NONE'}
                # выполнение сделки
                # Требования вызвавшие исключения:
                #   1) Минимальный размер ордера,
                #   2) Разрешение значения ордера (округление до 2-х знаков),
                #   3) Достаточность средств на счёте (продаваемая валюта).
                if predict == 1:  # продажа usdt, покупка btc
                    profit_fee_status = profit_status(btc_usdt_last_trade, ticker_price[0], fee, profit_fee_coef, 0) # btc_usdt_min_trade_count)
                    if trading_on == 1:  # проверка и изменение статуса только если он 1, 0 поменять нельзя (только вручную через БД)
                        trading_on = trading_status(ticker_price[0], trade_max_first, start, part, fee, profit_fee_coef, trade_move_coef)
                    if profit_fee_status and trading_on == 1:
                        print(volume_btc_buy, float(round((volume_btc_buy) , 4)))
                        order = client.new_order(type="MARKET", symbol="BTCUSDT", side="BUY", quantity=float(round((volume_btc_buy) , 4)))
                        # order = {'symbol': 'BTCUSDT', 'orderId': 0, 'orderListId': 0, 'clientOrderId': '0',
                        #          'transactTime': 0, 'price': '0.00000000', 'origQty': '0.00000000',
                        #          'executedQty': '0.00000000', 'cummulativeQuoteQty': '0', 'status': 'FILLED',
                        #          'timeInForce': 'GTC', 'type': 'None', 'side': 'None', 'workingTime': 0, 'fills': [
                        #         {'price': '0', 'qty': '0.000', 'commission': '0.00000', 'commissionAsset': 'None',
                        #          'tradeId': 0}], 'selfTradePreventionMode': 'NONE'}
                        btc_usdt_last_trade = ticker_price[0]  # сделать стоимость покупки из ордера
                        if btc_usdt_min_trade_count == 0:
                            trade_min_first = ticker_price[0]   #  цена первой сделка в серии min
                        btc_usdt_min_trade_count += 1
                        #btc_usdt_min_trade_count = 0
                        btc_usdt_max_trade_count = 0

                elif predict == 2:  # продажа btc, покупка usdt
                    profit_fee_status = profit_status(ticker_price[0], btc_usdt_last_trade, fee, profit_fee_coef, 0) # btc_usdt_max_trade_count)
                    if trading_on == 1:  # проверка и изменение статуса только если он 1, 0 поменять нельзя (только вручную через БД)
                        trading_on = trading_status(trade_min_first, ticker_price[0], start, part, fee, profit_fee_coef, trade_move_coef)
                    if profit_fee_status and trading_on == 1:
                        print(volume_usdt_buy, ticker_price[0], float(round((volume_usdt_buy / ticker_price[0]) , 4)))
                        order = client.new_order(type="MARKET", symbol="BTCUSDT", side="SELL", quantity=float(round((volume_usdt_buy / ticker_price[0]) , 4)))
                        # order = {'symbol': 'BTCUSDT', 'orderId': 0, 'orderListId': 0, 'clientOrderId': '0',
                        #          'transactTime': 0, 'price': '0.00000000', 'origQty': '0.00000000',
                        #          'executedQty': '0.00000000', 'cummulativeQuoteQty': '0', 'status': 'FILLED',
                        #          'timeInForce': 'GTC', 'type': 'None', 'side': 'None', 'workingTime': 0, 'fills': [
                        #         {'price': '0', 'qty': '0.000', 'commission': '0.00000', 'commissionAsset': 'None',
                        #          'tradeId': 0}], 'selfTradePreventionMode': 'NONE'}
                        btc_usdt_last_trade = ticker_price[0]  # сделать стоимость покупки из ордера
                        if btc_usdt_max_trade_count == 0:
                            trade_max_first = ticker_price[0]  #  цена первой сделка в серии max
                        btc_usdt_min_trade_count = 0
                        btc_usdt_max_trade_count += 1
                        #btc_usdt_max_trade_count = 0

                else:
                    order = {'symbol': 'None', 'orderId': 0, 'orderListId': 0, 'clientOrderId': '0', 'transactTime': 0, 'price': '0.00000000', 'origQty': '0.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'None', 'side': 'None', 'workingTime': 0, 'fills': [{'price': '0', 'qty': '0.000', 'commission': '0.00000', 'commissionAsset': 'None', 'tradeId': 0}], 'selfTradePreventionMode': 'NONE'}
            else:
                order = {'symbol': 'None', 'orderId': 0, 'orderListId': 0, 'clientOrderId': '0', 'transactTime': 0, 'price': '0.00000000', 'origQty': '0.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'None', 'side': 'None', 'workingTime': 0, 'fills': [{'price': '0', 'qty': '0.000', 'commission': '0.00000', 'commissionAsset': 'None', 'tradeId': 0}], 'selfTradePreventionMode': 'NONE'}

            # проверка отсутствия комиссии, в случае комиссии поставить trading_on в 0
            if float(order['fills'][0]['commission']) != 0:
                trading_on = 0
                sql_trading_settings_write(datetime.fromtimestamp(time()), trading_on, start, part, fee,
                                           profit_fee_coef, trade_move_coef)



            # пауза для 6-ти секундного периода
            sleep(4)

            # запись данных в БД binance
            sql_status = sql_trade_write(datetime.fromtimestamp(time_begin), time() - time_begin,
                                         predict, volume_status, profit_fee_status, min_order_status, volume_before,
                                         fee_usdt, part, slip_n, profit_fee_coef, btc_usdt_min_trade_count,
                                         btc_usdt_max_trade_count, ticker_price[3], ticker_price[4],
                                         ticker_price[5], ticker_price[0], ticker_price[1],
                                         ticker_price[2], coins[0]['free'], coins[1]['free'],
                                         volume_btc_buy, volume_usdt_buy,
                                         volume_btc_sell, volume_usdt_sell, order['orderId'],
                                         order['symbol'], order['fills'][0]['price'], order['fills'][0]['commission'],
                                         order['fills'][0]['commissionAsset'])

            # вывод информации на экран
            print('----')
            print(f'time: {datetime.fromtimestamp(time_begin)}, predict: {predict}, trade: {volume_status, profit_fee_status, min_order_status}, profit_last_600:_____, order: {order["symbol"]}')
            print(f'trading_on: {trading_on}, part: {part}, fee: {fee}, profit_fee_coef: {profit_fee_coef}, trade_move_coef: {trade_move_coef}')
        else:
            print(len(tickers_prices))
        tickers_prices = tickers_prices[-features_length+1:]



if __name__ == '__main__':
    main()