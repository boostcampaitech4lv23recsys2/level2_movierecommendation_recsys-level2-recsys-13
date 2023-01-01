import os
import random
import numpy as np
import torch
import pandas as pd

# seed


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


# data format preprocessing


def make_atomic(args, df: pd.DataFrame, name: str = 'recbole', dtype: str = 'inter'):
    assert dtype == 'inter'

    if dtype == 'inter':
        if os.path.exists(args.atomic_dir+'recbole.inter'):
            return
        df = convert_to_atomic(df)

    save_atomic(args, df, name, dtype)


def convert_to_atomic(df: pd.DataFrame) -> pd.DataFrame:
    """
    token, token_seq, float, float_seq
    """
    df['user'] = df['user'].astype(str)
    df['item'] = df['item'].astype(str)

    return df.rename(columns={'user': 'user:token', 'item': 'item:token'})


def save_atomic(args, df: pd.DataFrame, name: str, dtype: str):
    df.to_csv(args.atomic_dir + name + "." + dtype, index=False, sep='\t')


def make_sequential_atomic(args):
    if os.path.exists(args.atomic_dir + 'sequential.inter'):
        return
    
    df = pd.read_csv(args.data_dir + 'train_ratings.csv')
    df['user'] = df['user'].astype(str)
    df['item'] = df['item'].astype(str)
    df['time'] = df['time'].astype(str)
    
    df = df.rename(columns={'user': 'user:token', 'item': 'item:token', 'time': 'time:float'})
    
    df.to_csv(args.atomic_dir + args.dataset + ".inter", index=False, sep='\t')
    

# model