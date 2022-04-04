import pandas as pd
from tqdm import tqdm
import numpy as np

from data_generation.scheme import *


def create_dataset(schemes, count, name):
    min_x = 0
    max_x = 100

    df = pd.DataFrame(columns=['x0_mean', 'x0_std',
                               'x1_mean', 'x1_std',
                               'x2_mean', 'x2_std',
                               'x3_mean', 'x3_std',
                               'x4_mean', 'x4_std',
                               'x5_mean', 'x5_std',
                               'x6_mean', 'x6_std',
                               'x7_mean', 'x7_std',
                               'x8_mean', 'x8_std',
                               'x9_mean', 'x9_std',
                               'scheme'])

    for scheme in schemes:
        for _ in tqdm(range(count), f'{scheme}', leave=False):
            min_x_i = random.randint(min_x, max_x // 2)
            max_x_i = random.randint((min_x_i + max_x) // 2, max_x)

            positions = gen_scheme(scheme, min_x_i, max_x_i, series_time=10)
            random.shuffle(positions)

            data = [(np.mean(x), np.std(x)) for x in positions]
            data = [item for sublist in data for item in sublist]

            df.loc[len(df)] = data + [scheme]

    df.to_csv(name, index=False)


if __name__ == '__main__':
    schemes = [  # [3, 1, 4, 2],
        # [3, 4, 1, 2],
        # [3, 4, 2, 1],
        [3, 4, 3],
        # [3, 5, 1, 1],
        [3, 5, 2],
        # [4, 1, 2, 1, 2],
        # [4, 1, 3, 2],
        # [4, 1, 4, 1],
        [4, 2, 2, 2],
        # [4, 2, 3, 1],
        [4, 2, 4],
        # [4, 3, 1, 2],
        # [4, 3, 2, 1],
        [4, 3, 3],
        # [4, 4, 1, 1],
        [4, 4, 2],
        [4, 5, 1],
        # [5, 2, 1, 2],
        # [5, 2, 2, 1],
        [5, 2, 3],
        [5, 3, 2],
        [5, 4, 1]]

    print('Creating train.csv')
    create_dataset(schemes, 2, '../../data/series_train.csv')
    print('Creating test.csv')
    create_dataset(schemes, 1, '../../data/series_test.csv')
