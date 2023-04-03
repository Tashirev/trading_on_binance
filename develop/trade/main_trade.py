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
        self.bn1 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
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
        x = self.bn1(x)
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

def sell_buy(volume_x, volume_y, x_y_price, y_usdt_price, part, FEE, SLIP_N):  # продаваемая валюта X, покупаемая валюта Y, ....
    volume_x_sell = volume_x * part  # продаваемый объём в крипте
    volume_y_buy = volume_x_sell * x_y_price  # покупаемый объём в крипте
    fee = volume_y_buy * FEE * SLIP_N  # размер комиссии в крипте
    return volume_x_sell, volume_y_buy, fee * y_usdt_price  # продаваемый объём в крипте, покупаемый объём в крипте, комиссия в usdt
    # return volume_x - volume_x_sell, volume_y + volume_y_buy - fee, fee * y_usdt_price

# расчёт прибыли в usdt при сделке
def profit_calculation(coins, ticker_price, predict, part, fee, slip_n, profit_fee_coef):
    volume_btc = float(coins[0]['free'])
    volume_eth = float(coins[1]['free'])
    volume_bnb = float(coins[2]['free'])
    volume_bnb_buy = 0
    volume_btc_buy = 0
    volume_eth_buy = 0
    volume_bnb_sell = 0
    volume_btc_sell = 0
    volume_eth_sell = 0
    volume_before = volume_btc * ticker_price[0] + volume_eth * ticker_price[1] + volume_bnb * ticker_price[2]
    fee_usdt = 0
    volume_sell_crypt_after_usdt = 0 # остаток продаваемой крипты в usdt после сделки
    volume_status = 0  # статус не превышения минимально объёма крипты в кошельке
    profit_fee_status = 0 # статус превышения прибыли комиссии
    min_order_status = 0  # статус превышения минимально допустимого размера ордера Binance
    if predict == 1:   # продажа btc, покупка bnb
        volume_btc_sell, volume_bnb_buy, fee_usdt = sell_buy(volume_btc, volume_bnb, 1 / ticker_price[3], ticker_price[2], part, fee, slip_n)
        volume_sell_crypt_after_usdt = (volume_btc - volume_btc_sell) * ticker_price[0]
        volume_btc = volume_btc - volume_btc_sell
        volume_bnb = volume_bnb + volume_bnb_buy * (1 - fee * slip_n)
        if volume_btc_sell > 0.00015:# статус превышения минимально допустимого ордера Binance 0.0001 BTC
            min_order_status = 1
    elif predict == 2:   # продажа bnb, покупка btc
        volume_bnb_sell, volume_btc_buy, fee_usdt = sell_buy(volume_bnb, volume_btc, ticker_price[3], ticker_price[0], part, fee, slip_n)
        volume_sell_crypt_after_usdt = (volume_bnb - volume_bnb_sell) * ticker_price[2]
        volume_bnb = volume_bnb - volume_bnb_sell
        volume_btc = volume_btc + volume_btc_buy * (1 - fee * slip_n)
        if volume_btc_buy > 0.00015:# статус превышения минимально допустимого ордера Binance 0.0001 BTC
            min_order_status = 1
    elif predict == 3:   # продажа btc, покупка eth
        volume_btc_sell, volume_eth_buy, fee_usdt = sell_buy(volume_btc, volume_eth, 1 / ticker_price[4], ticker_price[1], part, fee, slip_n)
        volume_sell_crypt_after_usdt = (volume_btc - volume_btc_sell) * ticker_price[0]
        volume_btc = volume_btc - volume_btc_sell
        volume_eth = volume_eth + volume_eth_buy * (1 - fee * slip_n)
        if volume_btc_sell > 0.00015:# статус превышения минимально допустимого ордера Binance 0.0001 BTC
            min_order_status = 1
    elif predict == 4:  # продажа eth, покупка btc
        volume_eth_sell, volume_btc_buy, fee_usdt = sell_buy(volume_eth, volume_btc, ticker_price[4], ticker_price[0], part, fee, slip_n)
        volume_sell_crypt_after_usdt = (volume_eth - volume_eth_sell) * ticker_price[1]
        volume_eth = volume_eth - volume_eth_sell
        volume_btc = volume_btc + volume_btc_buy * (1 - fee * slip_n)
        if volume_btc_buy > 0.00015:# статус превышения минимально допустимого ордера Binance 0.0001 BTC
            min_order_status = 1
    elif predict == 5:  # продажа eth, покупка bnb
        volume_eth_sell, volume_bnb_buy, fee_usdt = sell_buy(volume_eth, volume_bnb, 1 / ticker_price[5], ticker_price[2], part, fee, slip_n)
        volume_sell_crypt_after_usdt = (volume_eth - volume_eth_sell) * ticker_price[1]
        volume_eth = volume_eth - volume_eth_sell
        volume_bnb = volume_bnb + volume_bnb_buy * (1 - fee * slip_n)
        if volume_eth_sell > 0.006:# статус превышения минимально допустимого ордера Binance 0.005 ETH
            min_order_status = 1
    elif predict == 6:  # продажа bnb, покупка eth
        volume_bnb_sell, volume_eth_buy, fee_usdt = sell_buy(volume_bnb, volume_eth, ticker_price[5], ticker_price[1], part, fee, slip_n)
        volume_sell_crypt_after_usdt = (volume_bnb - volume_bnb_sell) * ticker_price[2]
        volume_bnb = volume_bnb - volume_bnb_sell
        volume_eth = volume_eth + volume_eth_buy * (1 - fee * slip_n)
        if volume_eth_buy > 0.006:    # статус превышения минимально допустимого ордера Binance 0.005 ETH
            min_order_status = 1

    if volume_sell_crypt_after_usdt > 1:  # статус не превышения минимально объёма крипты в кошельке в долларах
        volume_status = 1

    volume_after = volume_btc * ticker_price[0] + volume_eth * ticker_price[1] + volume_bnb * ticker_price[2]
    profit_usdt = volume_after - volume_before

    if profit_usdt > fee_usdt * profit_fee_coef:  # условие превышения комиссии от сделки
        profit_fee_status = 1

    return volume_before, profit_usdt, fee_usdt, profit_fee_status, min_order_status, volume_status, volume_bnb_buy, volume_btc_buy, volume_eth_buy, volume_bnb_sell, volume_btc_sell, volume_eth_sell   #прибыли и комиссия в usdt


def main():
    print('Binance trade')

    random, image_length, features_length, batch_size, epoch, fee, slip_n = config_model()
    profit_fee_coef = 1  # для условия совершения сделки, доля комиссии для превышения профита (учёт проскальзывания, 0 - отсутствует проскальзывание)
    part = 0.3
    fee = 0.00075
    slip_n = 1
    features_length = 80
    tickers_prices = list()
    res_net = torch.load('res_net', map_location=torch.device('cpu')).eval()

    while True:

        time_begin = time()


        # запрос данных из binance
        coins, ticker_price = binance_read()
        tickers_prices.append(ticker_price)
        #print(coins)

        # подготовка features
        if(len(tickers_prices) >= features_length):
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
            volume_before, profit_usdt, fee_usdt, profit_fee_status, min_order_status, volume_status, volume_bnb_buy, volume_btc_buy, volume_eth_buy, volume_bnb_sell, volume_btc_sell, volume_eth_sell = profit_calculation(coins, ticker_price, predict, part, fee, slip_n, profit_fee_coef)


            # принятие решения по сделке и сделка
            if profit_fee_status + min_order_status + volume_status == 3: # сделка подходит по всем параметрам, все статусы 1
                # order = {'symbol': 'None', 'orderId': 999, 'orderListId': 0, 'clientOrderId': '0', 'transactTime': 0, 'price': '0.00000000', 'origQty': '0.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'None', 'side': 'None', 'workingTime': 0, 'fills': [{'price': '0', 'qty': '0.000', 'commission': '0.00000', 'commissionAsset': 'None', 'tradeId': 0}], 'selfTradePreventionMode': 'NONE'}
                # выполнение сделки
                # Требования вызвавшие исключения:
                #   1) Минимальный размер ордера,
                #   2) Разрешение значения ордера (округление до 2-х знаков),
                #   3) Достаточность средств на счёте (продаваемая валюта).
                if predict == 1:  # продажа btc, покупка bnb
                    print(volume_bnb_buy, float(round((volume_bnb_buy) , 3)))
                    order = client.new_order(type="MARKET", symbol="BNBBTC", side="BUY", quantity=float(round((volume_bnb_buy) , 3)))
                elif predict == 2:  # продажа bnb, покупка btc
                    print(volume_btc_buy, ticker_price[3], float(round((volume_btc_buy / ticker_price[3]) , 3)))
                    order = client.new_order(type="MARKET", symbol="BNBBTC", side="SELL", quantity=float(round((volume_btc_buy / ticker_price[3]) , 3)))
                elif predict == 3:  # продажа btc, покупка eth
                    print(volume_eth_buy, float(round((volume_eth_buy) , 4)))
                    order = client.new_order(type="MARKET", symbol="ETHBTC", side="BUY", quantity=float(round((volume_eth_buy) , 4)))
                elif predict == 4:  # продажа eth, покупка btc
                    print(volume_btc_buy, ticker_price[4], float(round((volume_btc_buy / ticker_price[4]) , 4)))
                    order = client.new_order(type="MARKET", symbol="ETHBTC", side="SELL", quantity=float(round((volume_btc_buy / ticker_price[4]) , 4)))
                elif predict == 5:  # продажа eth, покупка bnb
                    print(volume_bnb_buy, float(round((volume_bnb_buy) , 3)))
                    order = client.new_order(type="MARKET", symbol="BNBETH", side="BUY", quantity=float(round((volume_bnb_buy) , 3)))
                elif predict == 6:  # продажа bnb, покупка eth
                    print( volume_eth_buy, ticker_price[5],float(round((volume_eth_buy / ticker_price[5]) , 3)))
                    order = client.new_order(type="MARKET", symbol="BNBETH", side="SELL", quantity=float(round((volume_eth_buy / ticker_price[5]) , 3)))
                else:
                    order = {'symbol': 'None', 'orderId': 0, 'orderListId': 0, 'clientOrderId': '0', 'transactTime': 0, 'price': '0.00000000', 'origQty': '0.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'None', 'side': 'None', 'workingTime': 0, 'fills': [{'price': '0', 'qty': '0.000', 'commission': '0.00000', 'commissionAsset': 'None', 'tradeId': 0}], 'selfTradePreventionMode': 'NONE'}
            else:
                order = {'symbol': 'None', 'orderId': 0, 'orderListId': 0, 'clientOrderId': '0', 'transactTime': 0, 'price': '0.00000000', 'origQty': '0.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'None', 'side': 'None', 'workingTime': 0, 'fills': [{'price': '0', 'qty': '0.000', 'commission': '0.00000', 'commissionAsset': 'None', 'tradeId': 0}], 'selfTradePreventionMode': 'NONE'}

            # пауза для 6-ти секундного периода
            sleep(4)

            # запись данных в БД binance
            sql_status = sql_trade_write(datetime.fromtimestamp(time_begin), time() - time_begin,
                                         predict, volume_status, profit_fee_status, min_order_status, volume_before, profit_usdt,
                                         fee_usdt, part, slip_n, profit_fee_coef, ticker_price[3], ticker_price[4],
                                         ticker_price[5], ticker_price[0], ticker_price[1],
                                         ticker_price[2], coins[0]['free'], coins[1]['free'], coins[2]['free'],
                                         volume_btc_buy, volume_eth_buy, volume_bnb_buy,
                                         volume_btc_sell, volume_eth_sell, volume_bnb_sell, order['orderId'],
                                         order['symbol'], order['fills'][0]['price'], order['fills'][0]['commission'],
                                         order['fills'][0]['commissionAsset'])

            # вывод информации на экран
            print(f'time: {datetime.fromtimestamp(time_begin)}, predict: {predict}, trade: {volume_status, profit_fee_status, min_order_status}, profit: {profit_usdt}, order: {order["symbol"]}')
        else:
            print(len(tickers_prices))
        tickers_prices = tickers_prices[-79:]



if __name__ == '__main__':
    main()