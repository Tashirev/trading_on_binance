# Чтение файла конфигурации

import json
import os

def config(service):
    basedir = os.path.abspath(os.getcwd())
    workbooks_dir = os.path.abspath(os.path.join(basedir, '..'))
    with open(os.path.join(workbooks_dir, 'config_connect.json'), 'r') as f:
        config = json.load(f)
    if service == 'binance':
        config = config['binance']
    if service == 'postgres':
        config = config['postgres']

    return config