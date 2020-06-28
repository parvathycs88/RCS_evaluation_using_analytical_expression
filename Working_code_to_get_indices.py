import pandas as pd
import numpy as np


def search_coordinate(df_data: pd.DataFrame, search_set: set) -> list:
    nda_values = df_data.values
    tuple_index = np.where(np.isin(nda_values, [e for e in search_set]))
    return [(row, col, nda_values[row][col]) for row, col in zip(tuple_index[0], tuple_index[1])]

import decimal

def float_range(start, stop, step): # function to get range of floating point numbers
  while start < stop:
    yield float(start)
    start += decimal.Decimal(step)

#print(list(float_range(2, 4, '0.1600')))


if __name__ == '__main__':
    df_data = df2
    result_list = search_coordinate(df_data, list(float_range(2, 4, '0.1600')))
    print(f"\n\n{'row':<4} {'col':<4} {'name':>10}")
    [print(f"{row:<4} {col:<4} {name:>10}") for row, col, name in result_list]
