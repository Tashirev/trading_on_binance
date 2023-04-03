from binance.spot import Spot

client = Spot(key='AFohhuOqRYam933hqDEVWQsShwAReyuA7udzTlrxfCIpYSOQtuwvxi2Zs5iPYyjB',
              secret='RGr9OmZZuDGPedbk59g8jc9bJQUG4HT0RLfeT5PjX3dHZVLo0C5IJIzfhE4Jswrp')
pairs_binance = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "BNBBTC", "ETHBTC", "BNBETH"]
pairs = ['btc_usdt', 'eth_usdt', 'bnb_usdt', 'bnb_btc', 'eth_btc', 'bnb_eth']

account = client.account()
balances = account['balances']

for balanc in balances:
    if float(balanc['free']) != 0:
        print(balanc)

print('before trade')

#order = client.new_order(type="MARKET", symbol="BNBUSDT", side="BUY", quantity=0.11)
#order = client.new_order(type="MARKET", symbol="ETHUSDT", side="BUY", quantity=0.0079)
#order = client.new_order(type="MARKET", symbol="BTCUSDT", side="BUY", quantity=0.0016)

#print(order)
print('after trade')

balances = account['balances']
for balanc in balances:
    if float(balanc['free']) != 0:
        print(balanc)
