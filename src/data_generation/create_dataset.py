import pandas as pd
from tqdm import tqdm

from data_generation.scheme import *


def create_dataset(schemes, count, name):
    min_x = 0
    max_x = 100

    df = pd.DataFrame(columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'scheme'])

    for scheme in schemes:
        for _ in tqdm(range(count), f'{scheme}', leave=False):
            min_x_i = random.randint(min_x, max_x // 2)
            max_x_i = random.randint((min_x_i + max_x) // 2, max_x)

            positions = gen_scheme(scheme, min_x_i, max_x_i)
            random.shuffle(positions)

            df.loc[len(df)] = positions + [scheme]

    df.to_csv(name, index=False)
    # plot_scheme(df.loc[5]['positions'])


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
    create_dataset(schemes, 3000, '../../data/train.csv')
    print('Creating test.csv')
    create_dataset(schemes, 1000, '../../data/test.csv')
