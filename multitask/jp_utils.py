import os
import pandas as pd
import torch
from tb_utils import get_event_files, get_scalar_dataframe

def benefit(df1, df2, use_max=False):
    if use_max:
        df = df2.max() - df1.max()
    else:
        df = df1.min() - df2.min()
       
    return pd.DataFrame(df).T


def filter_dataframe(df, **filters):
    filtered = df
    for key, value in filters.items():
        mask = filtered[key] == value
        filtered = filtered[mask]
    return filtered


def show_differing(df):
    differing_columns = [col for col in df.columns if len(df[col].unique()) > 1]
    return df[differing_columns].reset_index(drop=True, inplace=False)


def remove_columns(df1, df2):
    slices = []
    for nm in df2.columns.names:
        vals = df1.columns.get_level_values(nm).unique()
        slices.append(vals.to_list())
    if isinstance(df2.columns, pd.MultiIndex):
        return df2.sort_index(1).loc[:, tuple(slices)]
    else:
        return df2.sort_index(1).loc[:, slices[0]]


def benefit_generalized(df1, df2, use_max=False):
    if use_max:
        df = df2.max().sub(df1.max(), axis="index")
    else:
        df = df1.min().sub(df2.min(), axis="index")

    return pd.DataFrame(df).T.reorder_levels(df1.columns.names, axis=1)


def get_args(directory, load=True, store=True):
    df_fname = os.path.join(directory, "args.csv")
    if load and os.path.isfile(df_fname):
        df = pd.read_csv(df_fname, index_col=0)
        return df

    # Little weird to tie this to event files, but I think it's what I want.
    event_files = get_event_files(directory)
    args = pd.DataFrame({ef: torch.load(os.path.join(os.path.dirname(ef), "args.pt")) for ef in event_files}).T

    if store:
        args.to_csv(df_fname)
    
    return args


def get_normalization_df(args, ntrain, value="validation-1/loss", **filters):
    args = filter_dataframe(args, ntrain=ntrain, single=True, **filters)
    event_files = args.index.to_list()
    dfs = {ef: get_scalar_dataframe(ef, maxlen=10000) for ef in event_files}
    event_files, dfs = filter_nones(event_files, dfs)
    names = ["single", "ntrain", "snr1", "scale1", "train_seed"]
    names = [n for n in names if n in args.columns]
    new_index = pd.MultiIndex.from_tuples([tuple(x) for x in args[names].values], names=names)
    df = pd.DataFrame({ef: dfs[ef][value] for ef in args.index}).set_axis(new_index, axis=1, inplace=False)
    
    return df


def filter_nones(event_files, dfs):
    new = list()
    for ef in event_files:
        df = dfs[ef]
        if df is None:
            dfs.pop(ef)
            continue
        new.append(ef)
    return new, dfs
