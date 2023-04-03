import utils.augmentation as aug
import numpy as np
from tqdm import tqdm


def aug_all(ticker_price, number_parts):

    ticker_price = np.expand_dims(ticker_price, axis=0)
    ticker_price_aug_list = list()
    for _ in tqdm(range(number_parts)):
        ticker_price_aug = np.zeros(ticker_price.shape)
        for i in range(6):
            # jitter
            ticker_price_aug[0, :, i] = aug.jitter(ticker_price[0, :, i], sigma=np.std(ticker_price[0, :, i]) / 2000)
            # scaling
            ticker_price_aug[0, :, i] = aug.scaling(ticker_price_aug[0, :, i], sigma=0.00005)
        # time_warp
        ticker_price_aug = aug.time_warp(ticker_price_aug, sigma=0.002, knot=100)
        ticker_price_aug = ticker_price_aug[0, :, :]
        ticker_price_aug_list.append(ticker_price_aug)

    ticker_price_aug = np.array(ticker_price_aug_list)
    ticker_price_aug = np.reshape(ticker_price_aug, (-1, 6))
    return ticker_price_aug
