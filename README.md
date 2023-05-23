# trading_on_binance
Набор микросервисов:
1) feature_download - сервис чтения трендов и объёмов с биржи Binance и сохранения в БД PostgreSQL.
<div id="header" align="left">
<img src="img\feature_download.jpg" width="600"/>
</div>

2) trade_btc_usdt - серовис торгового алгоритма в паре BTC/USDT
<div id="header" align="left">
<img src="img\trade_btc_usdt.jpg" width="800"/>
</div>

3) telegram - telegram-бот для оповещения и управления торговым процессом.
<div id="header" align="left">
<img src="img\telegram.jpg" width="800"/>
</div>

4) wallet - сервис чтения объёма кошелька в USDT на бирже Binance и сохранения в БД PostgreSQL.
<div id="header" align="left">
<img src="img\wallet.jpg" width="800"/>
</div>

5) trade (требуется переработка) - алгоритм торговли в 3 парах: BTC/ETH, BTC/BNB, BNB/ETH.
6) labels - разметка минимумов и максимумов на историческом тренде и сохранения в БД PostgreSQL.


