import numpy as np


def corr_df_to_distribution(df):
    """
    Function for transforming symmetric dataframe to list of values (only values above (below) the main diagonal)
    :param df: symmetric dataframe
    :return: list of values
    """
    c = 1
    corr = []
    for i, row in df.iterrows():
        for j in df.columns.tolist()[c:]:
            corr.append(row[j])

        c += 1
    return corr


def active_df_to_dict(df):
    """
    Function for transforming dataframe of active states to dict
    :param df: DataFrame of active states
    :return: dict with indexes of active states
    """
    d = {}
    for col in df:
        sig = df[col]
        active = sig[sig == True]

        idx = active.reset_index()[['index']].diff()
        idx = idx[idx['index'] > 1]

        d[col] = np.array_split(np.array(active.index.tolist()), idx.index.tolist())
    return d
