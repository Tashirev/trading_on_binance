# Чтение файлов конфигурации

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


def config_model():
    basedir = os.path.abspath(os.getcwd())
    workbooks_dir = os.path.abspath(os.path.join(basedir, '..'))
    with open(os.path.join(workbooks_dir, 'config_model.json'), 'r') as f:
        config_model = json.load(f)
    random = config_model['RANDOM']
    image_length = config_model['IMAGE_LENGTH']
    features_length = config_model['FEATURES_LENGTH']
    batch_size = config_model['BATCH_SIZE']
    epoch = config_model['EPOCH']
    fee = config_model['FEE']
    slip_n = config_model['SLIP_N']

    return random, image_length, features_length, batch_size, epoch, fee, slip_n
