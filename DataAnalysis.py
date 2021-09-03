# import scipy.stats as ss
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame as df
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


def test_csv():
    # csv_test = pd.read_csv('./인천광역시_연수구_문화재 현황_2020.05.03.csv', engine='python')
    # csv_test = pd.read_csv('./인천광역시_연수구_문화재 현황_2020.05.03.csv', encoding='utf-8')
    csv_test = pd.read_csv(
        "./인천광역시_연수구_문화재 현황_2020.05.03.csv", engine="python", encoding="cp949"
    )
    csv_test.shape
    csv_test.to_csv()


def test_df():
    data = {
        "ID": ["A1", "A2", "A3", "A4", "A5"],
        "X1": [1, 2, 3, 4, 5],
        "X2": [3.0, 4.5, 3.2, 4.0, 3.5],
    }

    data_df = DataFrame(data, index=["a", "b", "c", "d", "e"])
    print(f"df = [\n{data_df}\n]")

    data_df2 = data_df.reindex(["a", "b", "c", "d", "e", "f"])
    print(f"df = [\n{data_df2}\n]")

    data_df2.to_csv("./data_df2.csv", sep=",", na_rep="NaN")


def test_df_option():
    df_1 = df(
        data=np.arange(12).reshape(3, 4),
        index=["r0", "r1", "r2"],  # Will default to np.arange(n) if no indexing
        columns=["c0", "c1", "c2", "c3"],
        dtype="int",  # Data type to force, otherwise infer
        copy=False,  # Copy data from inputs
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
            "class_1": ["a", "a", "b", "b", "c"],
            "var_1": np.arange(5),
            "var_2": np.random.randn(5),
        },
        index=["r0", "r1", "r2", "r3", "r4"],
    )
    print(f"df_2 = [\n{df_2}\n]")
    print(f"df_2.index = [\n{df_2.index}\n]")

    # print(f"df_2.ix[2:] = [\n{df_2.ix[2:]}\n]")
    # print(f"df_2.ix[2] = [\n{df_2.ix[2]}\n]")

    d: df = df_2.loc["r2":"r3"]

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
    idx = ["r0", "r1", "r2", "r3", "r4"]
    df_1 = pd.DataFrame({"c1": np.arange(5), "c2": np.random.randn(5)}, index=idx)
    print(f"df_1 = [\n{df_1}\n]")

    new_idx = ["r0", "r1", "r2", "r5", "r6"]
    df_2 = df_1.reindex(new_idx)
    print(f"df_1 = [\n{df_1}\n]")
    print(f"df_2 = [\n{df_2}\n]")

    # 비어 있는 값 채우기
    df_3 = df_1.reindex(new_idx, fill_value=0)
    print(f"df_3 = [\n{df_3}\n]")

    # 비어 있는 값 채우기
    df_4 = df_1.reindex(new_idx, fill_value="missing")
    print(f"df_4 = [\n{df_4}\n]")

    # 비어 있는 값 채우기
    df_5 = df_1.reindex(new_idx, fill_value="NA")
    print(f"df_5 = [\n{df_5}\n]")


def test_df_date_index():
    date_idx = pd.date_range("11/27/2020", periods=5, freq="D")
    print(f"date_idx = [\n{date_idx}\n]")

    df_2 = pd.DataFrame({"c1": [10, 20, 30, 40, 50]}, index=date_idx)
    print(f"df_2 = [\n{df_2}\n]")

    date_idx_2 = pd.date_range("11/25/2020", periods=10, freq="D")
    df_3 = df_2.reindex(date_idx_2)
    print(f"df_3 = [\n{df_3}\n]")

    df_4 = df_2.reindex(date_idx_2, method="ffill")
    print(f"df_4 = [\n{df_4}\n]")

    df_5 = df_2.reindex(date_idx_2, method="bfill")
    print(f"df_5 = [\n{df_5}\n]")


def test_df_concat():
    df_1 = pd.DataFrame(
        {
            "A": ["A0", "A1", "A2"],
            "B": ["B0", "B1", "B2"],
            "C": ["C0", "C1", "C2"],
            "D": ["D0", "D1", "D2"],
        },
        index=[0, 1, 2],
    )
    print(f"df_1 = [\n{df_1}\n]")

    df_2 = pd.DataFrame(
        {
            "A": ["A3", "A4", "A5"],
            "B": ["B3", "B4", "B5"],
            "C": ["C3", "C4", "C5"],
            "D": ["D3", "D4", "D5"],
        },
        index=[3, 4, 5],
    )
    print(f"df_2 = [\n{df_2}\n]")

    df_12_axis0 = pd.concat([df_1, df_2])
    print(f"df_12_axis0 = [\n{df_12_axis0}\n]")

    df_3 = pd.DataFrame(
        {
            "E": ["A6", "A7", "A8"],
            "F": ["B6", "B7", "B8"],
            "G": ["C6", "C7", "C8"],
            "H": ["D6", "D7", "D8"],
        },
        index=[0, 1, 2],
    )
    print(f"df_1 = [\n{df_1}\n]")
    print(f"df_3 = [\n{df_3}\n]")

    df_13_axis1 = pd.concat([df_1, df_3], axis=1)
    print(f"df_13_axis1 = [\n{df_13_axis1}\n]")

    df_4 = pd.DataFrame(
        {
            "A": ["A0", "A1", "A2"],
            "B": ["B0", "B1", "B2"],
            "C": ["C0", "C1", "C2"],
            "E": ["E0", "E1", "E2"],
        },
        index=[0, 1, 3],
    )
    print(f"df_1 = [\n{df_1}\n]")
    print(f"df_4 = [\n{df_4}\n]")

    df_14_outer = pd.concat([df_1, df_4], join="outer")
    print(f"df_14_outer = [\n{df_14_outer}\n]")

    df_14_inner = pd.concat([df_1, df_4], join="inner")
    print(f"df_14_inner = [\n{df_14_inner}\n]")

    print(f"df_1 = [\n{df_1}\n]")
    print(f"df_4 = [\n{df_4}\n]")

    df_14_outer_axis1 = pd.concat([df_1, df_4], join="outer", axis=1)
    print(f"df_14_outer_axis1 = [\n{df_14_outer_axis1}\n]")

    df_14_inner_axis1 = pd.concat([df_1, df_4], join="inner", axis=1)
    print(f"df_14_inner_axis1 = [\n{df_14_inner_axis1}\n]")

    df_14_axis1_reindex = pd.concat([df_1, df_4], axis=1).reindex(df_1.index)
    print(f"df_14_axis1_reindex = [\n{df_14_axis1_reindex}\n]")

    print(f"\n")

    df_5 = pd.DataFrame(
        {
            "A": ["A0", "A1", "A2"],
            "B": ["B0", "B1", "B2"],
            "C": ["C0", "C1", "C2"],
            "D": ["D0", "D1", "D2"],
        },
        index=["r0", "r1", "r2"],
    )

    df_6 = pd.DataFrame(
        {
            "A": ["A3", "A4", "A5"],
            "B": ["B3", "B4", "B5"],
            "C": ["C3", "C4", "C5"],
            "D": ["D3", "D4", "D5"],
        },
        index=["r3", "r4", "r5"],
    )
    print(f"df_5 = [\n{df_5}\n]")
    print(f"df_6 = [\n{df_6}\n]")

    df_56_with_index = pd.concat([df_5, df_6], ignore_index=False)
    print(f"df_56_with_index = [\n{df_56_with_index}\n]")

    df_56_ignore_index = pd.concat([df_5, df_6], ignore_index=True)
    print(f"df_56_ignore_index = [\n{df_56_ignore_index}\n]")

    # 계층적 index
    df_56_with_keys = pd.concat([df_5, df_6], keys=["df_5", "df_6"])
    print(f"df_56_with_keys = [\n{df_56_with_keys}\n]")

    print(f"df_56_with_keys.loc['df_5'] = [\n{df_56_with_keys.loc['df_5']}\n]")
    print(
        f"df_56_with_keys.loc['df_5'][0:2] = [\n{df_56_with_keys.loc['df_5'][0:2]}\n]"
    )

    df_56_with_name = pd.concat(
        [df_5, df_6], keys=["df_5", "df_6"], names=["df_name", "row_number"]
    )
    print(f"df_56_with_name = [\n{df_56_with_name}\n]")
    print(f"\n")

    df_7 = pd.DataFrame(
        {
            "A": ["A0", "A1", "A2"],
            "B": ["B0", "B1", "B2"],
            "C": ["C0", "C1", "C2"],
            "D": ["D0", "D1", "D2"],
        },
        index=["r0", "r1", "r2"],
    )

    df_8 = pd.DataFrame(
        {
            "A": ["A2", "A3", "A4"],
            "B": ["B2", "B3", "B4"],
            "C": ["C2", "C3", "C4"],
            "D": ["D2", "D3", "D4"],
        },
        index=["r2", "r3", "r4"],
    )
    print(f"df_7 = [\n{df_7}\n]")
    print(f"df_8 = [\n{df_8}\n]")

    df_78_F_verify_integrity = pd.concat([df_7, df_8], verify_integrity=False)
    print(f"df_78_F_verify_integrity = [\n{df_78_F_verify_integrity}\n]")

    df_78_T_verify_integrity = pd.concat([df_7, df_8], verify_integrity=True)
    print(f"df_78_T_verify_integrity = [\n{df_78_T_verify_integrity}\n]")


def test_df_sr_concat():
    df_1 = pd.DataFrame(
        {
            "A": ["A0", "A1", "A2"],
            "B": ["B0", "B1", "B2"],
            "C": ["C0", "C1", "C2"],
            "D": ["D0", "D1", "D2"],
        },
        index=[0, 1, 2],
    )
    print(f"df_1 = [\n{df_1}\n]")

    Series_1 = pd.Series(["S1", "S2", "S3"], name="S")
    print(f"Series_1 = [\n{Series_1}\n]")

    df_1s = pd.concat([df_1, Series_1], axis=1)
    print(f"df_1s = [\n{df_1s}\n]")

    df_1s_ignore_index = pd.concat([df_1, Series_1], axis=1, ignore_index=True)
    print(f"df_1s_ignore_index = [\n{df_1s_ignore_index}\n]")

    Series_2 = pd.Series([0, 1, 2])  # without name
    Series_3 = pd.Series([3, 4, 5])  # without name
    print(f"Series_1 = [\n{Series_1}\n]")
    print(f"Series_2 = [\n{Series_2}\n]")
    print(f"Series_3 = [\n{Series_3}\n]")

    df_123 = pd.concat([Series_1, Series_2, Series_3], axis=1)
    print(f"df_123 = [\n{df_123}\n]")

    df_123_with_key = pd.concat(
        [Series_1, Series_2, Series_3], axis=1, keys=["C0", "C1", "C1"]
    )
    print(f"df_123_with_key = [\n{df_123_with_key}\n]")
    print(f"\n")

    Series_4 = pd.Series(["S1", "S2", "S3", "S4"], index=["A", "B", "C", "E"])
    print(f"df_1 = [\n{df_1}\n]")
    print(f"Series_4 = [\n{Series_4}\n]")

    df_1s_append = df_1.append(Series_4, ignore_index=True)
    print(f"df_1s_append = [\n{df_1s_append}\n]")


def test_df_merge():
    df_left = pd.DataFrame(
        {
            "KEY": ["K0", "K1", "K2", "K3"],
            "A": ["A0", "A1", "A2", "A3"],
            "B": ["B0", "B1", "B2", "B3"],
        }
    )

    df_right = pd.DataFrame(
        {
            "KEY": ["K2", "K3", "K4", "K5"],
            "C": ["C2", "C3", "C4", "C5"],
            "D": ["D2", "D3", "D4", "D5"],
        }
    )
    print(f"df_left  = [\n{df_left}\n]")
    print(f"df_right = [\n{df_right}\n]")

    df_merge_how_left = pd.merge(df_left, df_right, how="left", on="KEY")

    print(f"df_merge_how_left = [\n{df_merge_how_left}\n]")

    df_merge_how_right = pd.merge(df_left, df_right, how="right", on="KEY")
    print(f"df_merge_how_right = [\n{df_merge_how_right}\n]")

    df_merge_how_inner = pd.merge(df_left, df_right, how="inner", on="KEY")  # default
    print(f"df_merge_how_inner = [\n{df_merge_how_inner}\n]")

    df_merge_how_outer = pd.merge(df_left, df_right, how="outer", on="KEY")
    print(f"df_merge_how_outer = [\n{df_merge_how_outer}\n]")

    df_merge_how_outer_indicator = pd.merge(
        df_left,
        df_right,
        how="outer",
        on="KEY",
        indicator=True,
    )
    print(f"df_merge_how_outer_indicator = [\n{df_merge_how_outer_indicator}\n]")

    df_merge_how_outer_indicator_info = pd.merge(
        df_left,
        df_right,
        how="outer",
        on="KEY",
        indicator="indicator_info",
    )
    print(
        f"df_merge_how_outer_indicator_info = [\n{df_merge_how_outer_indicator_info}\n]"
    )
    print(f"\n")

    df_left_2 = pd.DataFrame(
        {
            "KEY": ["K0", "K1", "K2", "K3"],
            "A": ["A0", "A1", "A2", "A3"],
            "B": ["B0", "B1", "B2", "B3"],
            "C": ["C0", "C1", "C2", "C3"],
        }
    )

    df_right_2 = pd.DataFrame(
        {
            "KEY": ["K0", "K1", "K2", "K3"],
            "B": ["B0_2", "B1_2", "B2_2", "B3_2"],
            "C": ["C0_2", "C1_2", "C2_2", "C3_2"],
            "D": ["D0_2", "D1_2", "D2_2", "D3_3"],
        }
    )
    print(f"df_left_2 = [\n{df_left_2}\n]")
    print(f"df_right_2 = [\n{df_right_2}\n]")

    df_merge_suffix = pd.merge(
        df_left_2, df_right_2, how="inner", on="KEY", suffixes=("_left", "_right")
    )
    print(f"df_merge_suffix = [\n{df_merge_suffix}\n]")

    df_merge_default = pd.merge(df_left_2, df_right_2, how="inner", on="KEY")
    print(f"df_merge_default = [\n{df_merge_default}\n]")


def test_df_merge_index():
    df_left = pd.DataFrame(
        {"A": ["A0", "A1", "A2", "A3"], "B": ["B0", "B1", "B2", "B3"]},
        index=["K0", "K1", "K2", "K3"],
    )

    df_right = pd.DataFrame(
        {"C": ["C2", "C3", "C4", "C5"], "D": ["D2", "D3", "D4", "D5"]},
        index=["K2", "K3", "K4", "K5"],
    )
    print(f"df_left = [\n{df_left}\n]")
    print(f"df_right = [\n{df_right}\n]")

    df_merge_left_index = pd.merge(
        df_left, df_right, left_index=True, right_index=True, how="left"
    )
    print(f"df_merge_left_index = [\n{df_merge_left_index}\n]")

    df_merge_right_index = pd.merge(
        df_left, df_right, left_index=True, right_index=True, how="right"
    )
    print(f"df_merge_right_index = [\n{df_merge_right_index}\n]")

    df_join_right = df_left.join(df_right, how="right")
    print(f"df_join_right = [\n{df_join_right}\n]")

    df_merge_inner_join = pd.merge(
        df_left, df_right, left_index=True, right_index=True, how="inner"
    )
    print(f"df_merge_inner_join = [\n{df_merge_inner_join}\n]")

    df_inner_join = df_left.join(df_right, how="inner")
    print(f"df_inner_join = [\n{df_inner_join}\n]")

    df_merge_outer_join = pd.merge(
        df_left, df_right, left_index=True, right_index=True, how="outer"
    )
    print(f"df_merge_outer_join = [\n{df_merge_outer_join}\n]")

    df_outer_join = df_left.join(df_right, how="outer")
    print(f"df_outer_join = [\n{df_outer_join}\n]")

    df_left_2 = pd.DataFrame(
        {
            "KEY": ["K0", "K1", "K2", "K3"],
            "A": ["A0", "A1", "A2", "A3"],
            "B": ["B0", "B1", "B2", "B3"],
        }
    )

    df_right_2 = pd.DataFrame(
        {"C": ["C2", "C3", "C4", "C5"], "D": ["D2", "D3", "D4", "D5"]},
        index=["K2", "K3", "K4", "K5"],
    )
    print(f"df_left_2 = [\n{df_left_2}\n]")
    print(f"df_right_2 = [\n{df_right_2}\n]")

    df_merge_key_index_left = pd.merge(
        df_left_2, df_right_2, left_on="KEY", right_index=True, how="left"
    )
    print(f"df_merge_key_index_left = [\n{df_merge_key_index_left}\n]")


def test_df_isnull():
    df_left = pd.DataFrame(
        {
            "KEY": ["K0", "K1", "K2", "K3"],
            "A": ["A0", "A1", "A2", "A3"],
            "B": [0.5, 2.2, 3.6, 0.4],
        }
    )

    df_right = pd.DataFrame(
        {
            "KEY": ["K2", "K3", "K4", "K5"],
            "C": ["C2", "C3", "C4", "C5"],
            "D": ["D2", "D3", "D4", "D5"],
        }
    )
    print(f"df_left = [\n{df_left}\n]")
    print(f"df_right = [\n{df_right}\n]")

    df_all = pd.merge(df_left, df_right, how="outer", on="KEY")
    print(f"df_all = [\n{df_all}\n]")

    df_isnull_fn = pd.isnull(df_all)
    print(f"df_isnull_fn = [\n{df_isnull_fn}\n]")

    df_isnull_method = df_all.isnull()
    print(f"df_isnull_method = [\n{df_isnull_method}\n]")

    df_notnull_fn = pd.notnull(df_all)
    print(f"df_notnull_fn = [\n{df_notnull_fn}\n]")

    df_notnull_method = df_all.notnull()
    print(f"df_notnull_method = [\n{df_notnull_method}\n]")

    print(f"df_all = [\n{df_all}\n]")

    df_all.loc[[0, 1], "A":"B"] = None
    # df_all.loc[0:1, 'A':'B'] = None
    # df_all.loc[0:1, 'A'] = None
    # df_all.loc[0:1, 'B'] = None
    print(f"df_all = [\n{df_all}\n]")

    df_ab_isnull = df_all[["A", "B"]].isnull()
    print(f"df_ab_isnull = [\n{df_ab_isnull}\n]")

    df_isnull_cnt = df_all.isnull().sum()
    print(f"df_isnull_cnt = [\n{df_isnull_cnt}\n]")

    df_a_isnull_cnt = df_all["A"].isnull().sum()
    print(f"df_a_isnull_cnt = [\n{df_a_isnull_cnt}\n]")

    df_notnull_cnt = df_all.notnull().sum()
    print(f"df_notnull_cnt = [\n{df_notnull_cnt}\n]")

    print(f"df_all = [\n{df_all}\n]")
    df_all["NaN_cnt"] = df_all.isnull().sum(1)
    df_all["NotNull_cnt"] = df_all.notnull().sum(1)

    print(f"df_all = [\n{df_all}\n]")


def test_calc_null():
    df = pd.DataFrame(
        np.arange(10).reshape(5, 2),
        index=["a", "b", "c", "d", "e"],
        columns=["C1", "C2"],
    )
    print(f"df = [\n{df}\n]")

    df.loc[["b", "e"], ["C1"]] = None
    df.loc[["b", "c"], ["C2"]] = None

    print(f"df = [\n{df}\n]")

    print(f"df.sum() = [\n{df.sum()}\n]")
    print(f"df['C1'] = [\n{df['C1']}\n]")
    print(f"df['C1'].sum() = [\n{df['C1'].sum()}\n]")
    print(f"df['C1'].cumsum() = [\n{df['C1'].cumsum()}\n]")
    print(f"df.mean() = [\n{df.mean()}\n]")
    print(f"df.mean(1) = [\n{df.mean(1)}\n]")
    print(f"df.std() = [\n{df.std()}\n]")
    print(f"\n")
    print(f"df = [\n{df}\n]")

    df["C3"] = df["C1"] + df["C2"]
    print(f"df = [\n{df}\n]")

    df_2 = pd.DataFrame(
        {"C1": [1, 1, 1, 1, 1], "C4": [1, 1, 1, 1, 1]}, index=["a", "b", "c", "d", "e"]
    )
    print(f"df_2 = [\n{df_2}\n]")

    print(f"df + df_2 = [\n{df + df_2}\n]")


def test_fill_na():
    df = pd.DataFrame(np.random.randn(5, 3), columns=["C1", "C2", "C3"])
    print(f"df = [\n{df}\n]")

    df.iloc[0, 0] = None
    df.loc[1, ["C1", "C3"]] = None
    df.loc[2, ["C2"]] = np.nan
    df.loc[3, ["C2"]] = np.nan
    df.loc[4, ["C3"]] = np.nan

    print(f"df = [\n{df}\n]")

    df_0 = df.fillna(0)
    print(f"df_0 = [\n{df_0}\n]")

    df_missing = df.fillna("missing")
    print(f"df_missing = [\n{df_missing}\n]")

    print(f"df = [\n{df}\n]")

    print(f"df.fillna(method='ffill') = [\n{df.fillna(method='ffill')}\n]")
    print(f"df.fillna(method='pad') = [\n{df.fillna(method='pad')}\n]")
    print(f"df.fillna(method='bfill') = [\n{df.fillna(method='bfill')}\n]")
    print(
        f"df.fillna(method='ffill', limit=1) = [\n{df.fillna(method='ffill', limit=1)}\n]"
    )
    print(
        f"df.fillna(method='bfill', limit=1) = [\n{df.fillna(method='bfill', limit=1)}\n]"
    )

    print(f"df = [\n{df}\n]")
    print(f"df.mean() = [\n{df.mean()}\n]")
    print(f"df.fillna(df.mean()) = [\n{df.fillna(df.mean())}\n]")
    print(
        f"df.where(pd.notnull(df), df.mean(), axis='columns') = [\n{df.where(pd.notnull(df), df.mean(), axis='columns')}\n]"
    )

    print(f"df.mean()['C1':'C2'] = [\n{df.mean()['C1':'C2']}\n]")
    print(f"df.fillna(df.mean()['C1':'C2']) = [\n{df.fillna(df.mean()['C1':'C2'])}\n]")

    df_2 = pd.DataFrame({"C1": [1, 2, 3, 4, 5], "C2": [6, 7, 8, 9, 10]})
    print(f"df_2 = [\n{df_2}\n]")
    df_2.loc[[1, 3], ["C2"]] = np.nan
    print(f"df_2 = [\n{df_2}\n]")

    df_2["C2_New"] = np.where(pd.notnull(df_2["C2"]) == True, df_2["C2"], df_2["C1"])
    print(f"df_2 = [\n{df_2}\n]")

    for i in df_2.index:
        if pd.notnull(df_2.loc[i, "C2"]) == True:
            df_2.loc[i, "C2_New_2"] = df_2.loc[i, "C2"]
        else:
            df_2.loc[i, "C2_New_2"] = df_2.loc[i, "C1"]

    print(f"df_2 = [\n{df_2}\n]")


def test_drop_na():
    df = pd.DataFrame(np.random.randn(5, 4), columns=["C1", "C2", "C3", "C4"])
    print(f"df = [\n{df}\n]")

    df.loc[[0, 1], "C1"] = None
    df.loc[2, "C2"] = np.nan

    print(f"df = [\n{df}\n]")

    df_dop_row = df.dropna(axis=0)
    print(f"df_dop_row = [\n{df_dop_row}\n]")

    df_drop_column = df.dropna(axis=1)
    print(f"df_dop_column = [\n{df_drop_column}\n]")

    print(f"df = [\n{df}\n]")
    print(f"df['C1'].dropna() = [\n{df['C1'].dropna()}\n]")

    print(f"df[['C1', 'C2', 'C3']].dropna() = [\n{df[['C1', 'C2', 'C3']].dropna()}\n]")
    print(
        f"df[['C1', 'C2', 'C3']].dropna(axis=0) = [\n{df[['C1', 'C2', 'C3']].dropna(axis=0)}\n]"
    )
    print(
        f"df[['C1', 'C2', 'C3']].dropna(axis=1) = [\n{df[['C1', 'C2', 'C3']].dropna(axis=1)}\n]"
    )
    print(
        f"df.loc[[2,4],['C1', 'C2', 'C3']].dropna(axis=0) = [\n{df.loc[[2,4],['C1', 'C2', 'C3']].dropna(axis=0)}\n]"
    )


def test_interpolate():
    datestrs = ["12/1/2020", "12/03/2020", "12/04/2020", "12/10/2020"]
    dates = pd.to_datetime(datestrs)
    print(f"dates = [\n{dates}\n]")

    ts = pd.Series([1, np.nan, np.nan, 10], index=dates)
    print(f"ts = [\n{ts}\n]")

    ts_intp_linear = ts.interpolate()
    print(f"ts_intp_linear = [\n{ts_intp_linear}\n]")

    print(f"ts = [\n{ts}\n]")

    ts_intp_time = ts.interpolate(method="time")
    print(f"ts_intp_time = [\n{ts_intp_time}\n]")

    df = pd.DataFrame({"C1": [1, 2, np.nan, np.nan, 5], "C2": [6, 8, 10, np.nan, 20]})
    print(f"df = [\n{df}\n]")

    df_intp_values = df.interpolate(method="values")
    print(f"df_intp_values = [\n{df_intp_values}\n]")
    print(f"df.interpolate() = [\n{df.interpolate()}\n]")
    print(
        f"df.interpolate(method='values', limit=1) = [\n{df.interpolate(method='values', limit=1)}\n]"
    )
    print(
        f"df.interpolate(method='values', limit=1, limit_direction='backward') = [\n{df.interpolate(method='values', limit=1, limit_direction='backward')}\n]"
    )


def test_replace():
    ser = pd.Series([1, 2, 3, 4, np.nan])
    print(f"ser = [\n{ser}\n]")
    print(f"ser.replace(2, 20) = [\n{ser.replace(2, 20)}\n]")
    print(f"ser.replace(np.nan, 5) = [\n{ser.replace(np.nan, 5)}\n]")

    rep_map = ser.replace({1: 6, 2: 7, 3: 8, 4: 9, np.nan: 10})
    print(f"rep_map = [\n{rep_map}\n]")

    df = pd.DataFrame(
        {
            "C1": ["a_old", "b", "c", "d", "e"],
            "C2": [1, 2, 3, 4, 5],
            "C3": [6, 7, 8, 9, np.nan],
        }
    )

    print(f"df = [\n{df}\n]")

    df_rep = df.replace({"C1": "a_old"}, {"C1": "a_new"})
    print(f"df_rep = [\n{df_rep}\n]")

    print(
        f"df.replace({{'C3': np.nan}}, {{'C3': 10}}) = [\n{df.replace({'C3': np.nan}, {'C3': 10})}\n]"
    )


def test_dup():
    data = {
        "key1": ["a", "b", "b", "c", "c"],
        "key2": ["v", "w", "w", "x", "y"],
        "col": [1, 2, 3, 4, 5],
    }

    df = pd.DataFrame(data, columns=["key1", "key2", "col"])

    print(f"df = [\n{df}\n]")

    print(f"df.duplicated(['key1']) = [\n{df.duplicated(['key1'])}\n]")

    print(f"df.duplicated(['key1', 'key2']) = [\n{df.duplicated(['key1', 'key2'])}\n]")

    print(
        f"df.duplicated(['key1'], keep='last') = [\n{df.duplicated(['key1'], keep='last')}\n]"
    )

    print(
        f"df.duplicated(['key1'], keep=False) = [\n{df.duplicated(['key1'], keep=False)}\n]"
    )

    print(
        f"df.drop_duplicates(['key1'], keep='last') = [\n{df.drop_duplicates(['key1'], keep='last')}\n]"
    )


def test_df_unique():
    df = pd.DataFrame(
        {
            "A": ["A1", "A1", "A2", "A2", "A3", "A3"],
            "B": ["B1", "B1", "B1", "B1", "B2", np.nan],
            "C": [1, 1, 3, 4, 4, 4],
        }
    )

    print(f"df = [\n{df}\n]")
    print(f"df['A'].unique() = [\n{df['A'].unique()}\n]")
    print(f"df['B'].unique() = [\n{df['B'].unique()}\n]")
    print(f"df['C'].unique() = [\n{df['C'].unique()}\n]")
    print(f"\n")

    print(f"df = [\n{df}\n]")
    print(f"df['A'].value_counts() = [\n{df['A'].value_counts()}\n]")
    print(f"df['B'].value_counts() = [\n{df['B'].value_counts()}\n]")
    print(f"df['C'].value_counts() = [\n{df['C'].value_counts()}\n]")
    print(
        f"df['C'].value_counts(normalize=True) = [\n{df['C'].value_counts(normalize=True)}\n]"
    )
    print(f"\n")

    print(f"df = [\n{df}\n]")
    print(f"df['C'].value_counts(sort=True) = [\n{df['C'].value_counts(sort=True)}\n]")
    print(
        f"df['C'].value_counts(sort=True, ascending=True) = [\n{df['C'].value_counts(sort=True, ascending=True)}\n]"
    )
    print(
        f"df['C'].value_counts(sort=False) = [\n{df['C'].value_counts(sort=False)}\n]"
    )
    print(f"\n")

    print(f"df = [\n{df}\n]")
    print(
        f"df['B'].value_counts(dropna=True) = [\n{df['B'].value_counts(dropna=True)}\n]"
    )
    print(
        f"df['B'].value_counts(dropna=False) = [\n{df['B'].value_counts(dropna=False)}\n]"
    )
    print(f"\n")

    print(f"df = [\n{df}\n]")
    print(
        f"df['C'].value_counts(bins=[0, 1, 2, 3, 4, 5], sort=False) = [\n{df['C'].value_counts(bins=[0, 1, 2, 3, 4, 5], sort=False)}\n]"
    )

    out = pd.cut(df["C"], bins=[0, 1, 2, 3, 4, 5])
    print(f"out = [\n{out}\n]")
    print(f"pd.value_counts(out) = [\n{pd.value_counts(out)}\n]")


def test_standardization():
    print(f"\n")
    data = np.random.randint(30, size=(6, 5))
    print(f"data = [\n{data}\n]")

    # z = (x - Mean)/STD
    data_standadized_np = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    print(f"data_standadized_np = [\n{data_standadized_np}\n]")

    print(
        f"np.mean(data_standadized_np, axis=0) = [\n{np.mean(data_standadized_np, axis=0)}\n]"
    )
    print(
        f"np.std(data_standadized_np, axis=0) = [\n{np.std(data_standadized_np, axis=0)}\n]"
    )
    print(
        f"np.std(data_standadized_np, axis=1) = [\n{np.std(data_standadized_np, axis=1)}\n]"
    )

    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"arr = [\n{arr}\n]")
    print(f"np.mean(arr) = [\n{np.mean(arr)}\n]")
    print(f"np.std(arr) = [\n{np.std(arr)}\n]")
    print(f"np.var(arr) = [\n{np.var(arr)}\n]")

    # data_standadized_ss = ss.zscore(data)
    # print(f"data_standadized_ss = [\n{data_standadized_ss}\n]")


def test_01scale():
    X = np.array([[10.0, -10.0, 1.0], [5.0, 0.0, 2.0], [0.0, 10.0, 3.0]])

    print(f"X = [\n{X}\n]")

    print(f"X.min(axis=0)= [\n{X.min(axis=0)}\n]")
    print(f"X.max(axis=0)= [\n{X.max(axis=0)}\n]")
    print(f"(X.max(axis=0) - X.min(axis=0)) = [\n{(X.max(axis=0) - X.min(axis=0))}\n]")

    X_MinMax = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    print(f"X_MinMax = [\n{X_MinMax}\n]")

    X_MinMax_train = MinMaxScaler().fit_transform(X)
    print(f"X_MinMax_train = [\n{X_MinMax_train}\n]")

    X_new = np.array([[9.0, -10.0, 1.0], [5.0, -5.0, 3.0], [1.0, 0.0, 5.0]])
    print(f"X_new = [\n{X_new}\n]")

    X_new_MinMax_train = MinMaxScaler().fit_transform(X_new)
    print(f"X_new_MinMax_train = [\n{X_new_MinMax_train}\n]")

    # X_MinMax_scaled = MinMaxScaler(X, axis=0, copy=True)
    # X_MinMax_scaled = MinMaxScaler(X, copy=True)
    # print(f"X_MinMax_scaled = [\n{X_MinMax_scaled}\n]")


if __name__ == "__main__":
    # test_df()
    # test_df_option()
    # test_df_fill_value()
    # test_df_date_index()
    # test_df_concat()
    # test_df_sr_concat()
    # test_df_merge()
    # test_df_merge_index()
    # test_df_isnull()
    # test_calc_null()
    # test_fill_na()
    # test_drop_na()
    # test_interpolate()
    # test_replace()
    # test_dup()
    # test_df_unique()
    # test_standardization()
    test_01scale()
