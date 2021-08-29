import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame as df


def test_csv():
    #csv_test = pd.read_csv('./인천광역시_연수구_문화재 현황_2020.05.03.csv', engine='python')
    #csv_test = pd.read_csv('./인천광역시_연수구_문화재 현황_2020.05.03.csv', encoding='utf-8')
    csv_test = pd.read_csv('./인천광역시_연수구_문화재 현황_2020.05.03.csv', engine='python', encoding='cp949')
    csv_test.shape
    csv_test.to_csv()

def test_df():
    data = {'ID': ['A1', 'A2', 'A3', 'A4', 'A5'],
        'X1': [1, 2, 3, 4, 5],
        'X2': [3.0, 4.5, 3.2, 4.0, 3.5],
    }

    data_df = DataFrame(data, index=['a', 'b', 'c', 'd', 'e'])
    print(f"df = [\n{data_df}\n]")

    data_df2 = data_df.reindex(['a', 'b', 'c', 'd', 'e', 'f'])
    print(f"df = [\n{data_df2}\n]")

    data_df2.to_csv('./data_df2.csv', sep=',', na_rep='NaN')

def test_df_option():
    df_1 = df(
        data=np.arange(12).reshape(3, 4),
        index=['r0', 'r1', 'r2'], # Will default to np.arange(n) if no indexing
        columns=['c0', 'c1', 'c2', 'c3'],
        dtype='int', # Data type to force, otherwise infer
        copy=False, # Copy data from inputs
    )
    print(f"df_1 = [\n{df_1}\n]")
    print(f"df_1.T = [\n{df_1.T}\n]")
    print(f"df_1.axes = [\n{df_1.axes}\n]")
    print(f"df_1.dtypes = [\n{df_1.dtypes}\n]")
    print(f"df_1.shape = [\n{df_1.shape}\n]")
    print(f"df_1.size = [\n{df_1.size}\n]")
    print(f"df_1.values = [\n{df_1.values}\n]")

    df_2 = df(
        {
            'class_1': ['a', 'a', 'b', 'b', 'c'],
            'var_1': np.arange(5),
            'var_2': np.random.randn(5)
        },
        index = ['r0', 'r1', 'r2', 'r3', 'r4']
    )
    print(f"df_2 = [\n{df_2}\n]")
    print(f"df_2.index = [\n{df_2.index}\n]")

    #print(f"df_2.ix[2:] = [\n{df_2.ix[2:]}\n]")
    #print(f"df_2.ix[2] = [\n{df_2.ix[2]}\n]")

    d : df = df_2.loc['r2':'r3']

    print(f"df_2.iloc[2:] = [\n{df_2.iloc[2:]}\n]")
    print(f"df_2.iloc[2:3] = [\n{df_2.iloc[2:3]}\n]")
    print(f"df_2.iloc[2] = [\n{df_2.iloc[2]}\n]")
    print(f"df_2.loc['r2':'r4'] = [\n{df_2.loc['r2':'r4']}\n]")

    print(f"df_2.head(2) = [\n{df_2.head(2)}\n]")
    print(f"df_2.tail(2) = [\n{df_2.tail(2)}\n]")

    print(f"\n")

    print(f"df_2.columns = [\n{df_2.columns}\n]")
    print(f"df_2['class_1'] = [\n{df_2['class_1']}\n]")
    print(f"df_2[['class_1', 'var_1']] = [\n{df_2[['class_1', 'var_1']]}\n]")

def test_df_fill_value():
    idx = ['r0', 'r1', 'r2', 'r3', 'r4']
    df_1 = pd.DataFrame(
        {
            'c1': np.arange(5),
            'c2': np.random.randn(5)
        },
        index=idx
    )
    print(f"df_1 = [\n{df_1}\n]")

    new_idx = ['r0', 'r1', 'r2', 'r5', 'r6']
    df_2 = df_1.reindex(new_idx)
    print(f"df_1 = [\n{df_1}\n]")
    print(f"df_2 = [\n{df_2}\n]")

    # 비어 있는 값 채우기
    df_3 = df_1.reindex(new_idx, fill_value=0)
    print(f"df_3 = [\n{df_3}\n]")

    # 비어 있는 값 채우기
    df_4 = df_1.reindex(new_idx, fill_value='missing')
    print(f"df_4 = [\n{df_4}\n]")

    # 비어 있는 값 채우기
    df_5 = df_1.reindex(new_idx, fill_value='NA')
    print(f"df_5 = [\n{df_5}\n]")

def test_df_date_index():
    date_idx = pd.date_range('11/27/2020', periods=5, freq='D')
    print(f"date_idx = [\n{date_idx}\n]")

    df_2 = pd.DataFrame(
        {
            "c1": [10, 20, 30, 40, 50]
        },
        index=date_idx
    )
    print(f"df_2 = [\n{df_2}\n]")

    date_idx_2 = pd.date_range('11/25/2020', periods=10, freq='D')
    df_3 = df_2.reindex(date_idx_2)
    print(f"df_3 = [\n{df_3}\n]")

    df_4 = df_2.reindex(date_idx_2, method='ffill')
    print(f"df_4 = [\n{df_4}\n]")

    df_5 = df_2.reindex(date_idx_2, method='bfill')
    print(f"df_5 = [\n{df_5}\n]")


def test_df_concat():
    df_1 = pd.DataFrame(
        {
            'A': ['A0', 'A1', 'A2'],
            'B': ['B0', 'B1', 'B2'],
            'C': ['C0', 'C1', 'C2'],
            'D': ['D0', 'D1', 'D2']
        },
        index=[0, 1, 2]
    )
    print(f"df_1 = [\n{df_1}\n]")

    df_2 = pd.DataFrame(
        {
            'A': ['A3', 'A4', 'A5'],
            'B': ['B3', 'B4', 'B5'],
            'C': ['C3', 'C4', 'C5'],
            'D': ['D3', 'D4', 'D5']
        },
        index=[3, 4, 5]
    )
    print(f"df_2 = [\n{df_2}\n]")

    df_12_axis0 = pd.concat([df_1, df_2])
    print(f"df_12_axis0 = [\n{df_12_axis0}\n]")

    df_3 = pd.DataFrame(
        {
            'E': ['A6', 'A7', 'A8'],
            'F': ['B6', 'B7', 'B8'],
            'G': ['C6', 'C7', 'C8'],
            'H': ['D6', 'D7', 'D8']
        },
        index=[0, 1, 2]
    )
    print(f"df_1 = [\n{df_1}\n]")
    print(f"df_3 = [\n{df_3}\n]")

    df_13_axis1 = pd.concat([df_1, df_3], axis=1)
    print(f"df_13_axis1 = [\n{df_13_axis1}\n]")

    df_4 = pd.DataFrame(
        {
            'A': ['A0', 'A1', 'A2'],
            'B': ['B0', 'B1', 'B2'],
            'C': ['C0', 'C1', 'C2'],
            'E': ['E0', 'E1', 'E2']
        },
        index=[0, 1, 3]
    )
    print(f"df_1 = [\n{df_1}\n]")
    print(f"df_4 = [\n{df_4}\n]")

    df_14_outer = pd.concat([df_1, df_4], join='outer')
    print(f"df_14_outer = [\n{df_14_outer}\n]")

    df_14_inner = pd.concat([df_1, df_4], join='inner')
    print(f"df_14_inner = [\n{df_14_inner}\n]")

    print(f"df_1 = [\n{df_1}\n]")
    print(f"df_4 = [\n{df_4}\n]")

    df_14_outer_axis1 = pd.concat([df_1, df_4], join='outer', axis=1)
    print(f"df_14_outer_axis1 = [\n{df_14_outer_axis1}\n]")

    df_14_inner_axis1 = pd.concat([df_1, df_4], join='inner', axis=1)
    print(f"df_14_inner_axis1 = [\n{df_14_inner_axis1}\n]")

    df_14_axis1_reindex = pd.concat([df_1, df_4], axis=1).reindex(df_1.index)
    print(f"df_14_axis1_reindex = [\n{df_14_axis1_reindex}\n]")

    print(f"\n")

    df_5 = pd.DataFrame(
        {
            'A': ['A0', 'A1', 'A2'],
            'B': ['B0', 'B1', 'B2'],
            'C': ['C0', 'C1', 'C2'],
            'D': ['D0', 'D1', 'D2']
        },
        index=['r0', 'r1', 'r2']
    )

    df_6 = pd.DataFrame(
        {
            'A': ['A3', 'A4', 'A5'],
            'B': ['B3', 'B4', 'B5'],
            'C': ['C3', 'C4', 'C5'],
            'D': ['D3', 'D4', 'D5']
        },
        index=['r3', 'r4', 'r5']
    )
    print(f"df_5 = [\n{df_5}\n]")
    print(f"df_6 = [\n{df_6}\n]")

    df_56_with_index = pd.concat([df_5, df_6], ignore_index=False)
    print(f"df_56_with_index = [\n{df_56_with_index}\n]")

    df_56_ignore_index = pd.concat([df_5, df_6], ignore_index=True)
    print(f"df_56_ignore_index = [\n{df_56_ignore_index}\n]")

    # 계층적 index
    df_56_with_keys = pd.concat([df_5, df_6], keys=['df_5', 'df_6'])
    print(f"df_56_with_keys = [\n{df_56_with_keys}\n]")

    print(f"df_56_with_keys.loc['df_5'] = [\n{df_56_with_keys.loc['df_5']}\n]")
    print(f"df_56_with_keys.loc['df_5'][0:2] = [\n{df_56_with_keys.loc['df_5'][0:2]}\n]")

    df_56_with_name = pd.concat(
        [df_5, df_6],
        keys=['df_5', 'df_6'],
        names=['df_name', 'row_number']
    )
    print(f"df_56_with_name = [\n{df_56_with_name}\n]")
    print(f"\n")

    df_7 = pd.DataFrame(
        {
            'A': ['A0', 'A1', 'A2'],
            'B': ['B0', 'B1', 'B2'],
            'C': ['C0', 'C1', 'C2'],
            'D': ['D0', 'D1', 'D2']
        },
        index=['r0', 'r1', 'r2']
    )

    df_8 = pd.DataFrame(
        {
            'A': ['A2', 'A3', 'A4'],
            'B': ['B2', 'B3', 'B4'],
            'C': ['C2', 'C3', 'C4'],
            'D': ['D2', 'D3', 'D4']
        },
        index=['r2', 'r3', 'r4']
    )
    print(f"df_7 = [\n{df_7}\n]")
    print(f"df_8 = [\n{df_8}\n]")

    df_78_F_verify_integrity = pd.concat(
        [df_7, df_8], 
        verify_integrity=False
    )
    print(f"df_78_F_verify_integrity = [\n{df_78_F_verify_integrity}\n]")

    df_78_T_verify_integrity = pd.concat(
        [df_7, df_8], 
        verify_integrity=True
    )
    print(f"df_78_T_verify_integrity = [\n{df_78_T_verify_integrity}\n]")


def test_df_sr_concat():
    df_1 = pd.DataFrame(
        {
            'A': ['A0', 'A1', 'A2'],
            'B': ['B0', 'B1', 'B2'],
            'C': ['C0', 'C1', 'C2'],
            'D': ['D0', 'D1', 'D2']
        },
        index=[0, 1, 2]
    )
    print(f"df_1 = [\n{df_1}\n]")

    Series_1 = pd.Series(
        ['S1', 'S2', 'S3'],
        name='S'
    )
    print(f"Series_1 = [\n{Series_1}\n]")

    df_1s = pd.concat([df_1, Series_1], axis=1)
    print(f"df_1s = [\n{df_1s}\n]")

    df_1s_ignore_index = pd.concat(
        [df_1, Series_1], 
        axis=1, 
        ignore_index=True
    )
    print(f"df_1s_ignore_index = [\n{df_1s_ignore_index}\n]")

    Series_2 = pd.Series([0, 1, 2]) # without name
    Series_3 = pd.Series([3, 4, 5]) # without name
    print(f"Series_1 = [\n{Series_1}\n]")
    print(f"Series_2 = [\n{Series_2}\n]")
    print(f"Series_3 = [\n{Series_3}\n]")

    df_123 = pd.concat(
        [Series_1, Series_2, Series_3], 
        axis=1
    )
    print(f"df_123 = [\n{df_123}\n]")

    df_123_with_key = pd.concat(
        [Series_1, Series_2, Series_3], 
        axis=1, 
        keys=['C0', 'C1', 'C1']
    )
    print(f"df_123_with_key = [\n{df_123_with_key}\n]")
    print(f"\n")

    Series_4 = pd.Series(
        ['S1', 'S2', 'S3', 'S4'], 
        index=['A', 'B', 'C', 'E']
    )
    print(f"df_1 = [\n{df_1}\n]")
    print(f"Series_4 = [\n{Series_4}\n]")

    df_1s_append = df_1.append(Series_4, ignore_index=True)
    print(f"df_1s_append = [\n{df_1s_append}\n]")



def test_df_merge():
    df_left = pd.DataFrame(
        {
            'KEY': ['K0', 'K1', 'K2', 'K3'],
            'A': ['A0', 'A1', 'A2', 'A3'],
            'B': ['B0', 'B1', 'B2', 'B3']
        }
    )

    df_right = pd.DataFrame(
        {
            'KEY': ['K2', 'K3', 'K4', 'K5'],
            'C': ['C2', 'C3', 'C4', 'C5'],
            'D': ['D2', 'D3', 'D4', 'D5']
        }
    )
    print(f"df_left  = [\n{df_left}\n]")
    print(f"df_right = [\n{df_right}\n]")

    df_merge_how_left = pd.merge(
        df_left, 
        df_right, 
        how='left', 
        on='KEY'
    )

    print(f"df_merge_how_left = [\n{df_merge_how_left}\n]")

    df_merge_how_right = pd.merge(
        df_left, 
        df_right, 
        how='right', 
        on='KEY'
    )
    print(f"df_merge_how_right = [\n{df_merge_how_right}\n]")

    df_merge_how_inner = pd.merge(
        df_left, 
        df_right, 
        how='inner', # default 
        on='KEY'
    )
    print(f"df_merge_how_inner = [\n{df_merge_how_inner}\n]")

    df_merge_how_outer = pd.merge(
        df_left, 
        df_right, 
        how='outer', 
        on='KEY'
    )
    print(f"df_merge_how_outer = [\n{df_merge_how_outer}\n]")

    df_merge_how_outer_indicator = pd.merge(
        df_left, 
        df_right, 
        how='outer', 
        on='KEY',
        indicator=True,
    )
    print(f"df_merge_how_outer_indicator = [\n{df_merge_how_outer_indicator}\n]")


if __name__ == '__main__':
    #test_df()
    #test_df_option()
    #test_df_fill_value()
    #test_df_date_index()
    #test_df_concat()
    #test_df_sr_concat()
    test_df_merge()