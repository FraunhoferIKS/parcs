import numpy as np
import pandas as pd

def dummy_data_csv(name=None, col=None, size=None):
    df = pd.DataFrame(
        np.random.normal(size=(size, col)),
        columns=['Z_{}'.format(i) for i in range(col)]
    )
    df.to_csv('./description_files/{}.csv'.format(name))

if __name__ == '__main__':
    dummy_data_csv('data_2', 3, 10)