from recbole.model.context_aware_recommender.fm import FM
from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset
from tqdm.notebook import tqdm

import pandas as pd
import numpy as np
import os
from os.path import join

def convert_to_atomic(df: pd.DataFrame, columns: dict = None) -> pd.DataFrame:
    """
    token, token_seq, float, float_seq
    """
    assert columns != None
    # change dtype to str
    for col in df.columns.tolist():
        df[col] = df[col].astype(str)
    # convert to atomic file format
    return df.rename(columns=columns)
def save_atomic(df: pd.DataFrame, name: str, dtype: str):
    df.to_csv(name+"."+dtype, index=False, sep='\t')

data_path   = './data/data2'
base        = 'base'
inter_df    = pd.read_csv(os.path.join(data_path, 'interactions.csv')) # 전체 학습 데이터
genre_df    = pd.read_csv(join(data_path, 'genres.csv'))
title_df    = pd.read_csv(join(data_path, 'titles.csv'))
director_df = pd.read_csv(join(data_path, 'directors.csv'))
writer_df   = pd.read_csv(join(data_path, 'writers.csv'))
year_df     = pd.read_csv(join(data_path, 'years.csv'))

# yaml
cfg_str = """
data_path: /opt/ml/input/data/
dataset: base
field_separator: "\\t"
seq_separator: "\\t"
USER_ID_FIELD: user
ITEM_ID_FIELD: item
TIME_FIELD: time
load_col:
  inter: [user, item, time]
  item: [item, year, title, genre, writer, director]
# model config
embedding_size: 10 # (int) The embedding size of features.
mlp_hidden_size: [64, 32, 16] # (list of int) The hidden size of MLP layers.
dropout_prob: 0.2 # (float) The dropout rate.
# Training and evaluation config
eval_setting: RO_RS,full
epochs: 10
seed: 42
train_batch_size: 4096
eval_batch_size: 4096
eval_args:
  split: { "RS": [0.4, 0.1, 0.1] }
  order: RO
  group_by: "user"
  mode: full
topk: 10
train_neg_sample_args:
  distribution: uniform
  sample_num: 1
  alpha: 1.0
  dynamic: False
  candidate_num: 0
metrics: ["Recall", "MRR", "NDCG", "Hit", "Precision"]
valid_metric: Recall@10
# logging
show_progress: true
"""
# yaml 파일 만들
yaml = '/opt/ml/input/data/base/base.yaml'
with open(yaml, "w") as f:
    f.write(cfg_str)
    
# dataset
inter_cols = {'user': 'user:token', 'item': 'item:token', 'time': 'time:float'}
atom_inter = convert_to_atomic(inter_df, inter_cols)
save_atomic(atom_inter, '/opt/ml/input/data/base/base', 'inter')

merged_df = genre_df
for df in [writer_df, director_df, year_df]:
    merged_df = pd.merge(merged_df, df, on='item', how='outer')
cols = ['genre', 'writer', 'director', 'year']
merged_df[cols] = merged_df[cols].astype(str)
merged_df['year'] = merged_df['year'].astype(float)

merged_df = (merged_df.groupby(['item', 'year'])
             .agg({'genre': lambda x: '\t'.join(x.tolist()),
                   'writer': lambda x: '\t'.join(x.tolist()),
                   'director': lambda x: '\t'.join(x.tolist()),
                   })
             .reset_index())
item_cols = {'item': 'item:token', 
        'genre': 'genre:token_seq',
        'year': 'year:float',
        'writer': 'writer:token_seq',
        'director': 'director:token_seq',}
atom_merged = convert_to_atomic(merged_df, item_cols)
save_atomic(atom_merged, '/opt/ml/input/data/base/base', 'item')

# run
run_recbole(
            model='LR',
            dataset=base,
            config_file_list=[yaml],
            )

