from recbole.model.context_aware_recommender.fm import FM
from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset
from tqdm.notebook import tqdm

from utils import *

context = False
MODEL = 'FM'
version = 'v2'
data_path = f'/opt/ml/input/data/train'

base = 'base'
atomic_path = f'/opt/ml/input/data/base/base'
yaml_path = f'/opt/ml/input/data/base/base.yaml'
inter_df = pd.read_csv(os.path.join(
    data_path, 'interactions.csv'))  # 전체 학습 데이터
genre_df = pd.read_csv(join(data_path, 'genres.csv'))
title_df = pd.read_csv(join(data_path, 'titles.csv'))
director_df = pd.read_csv(join(data_path, 'directors.csv'))
writer_df = pd.read_csv(join(data_path, 'writers.csv'))
year_df = pd.read_csv(join(data_path, 'years.csv'))

# yaml
cfg_str = """

data_path: /opt/ml/input/data/
dataset: v2_base
field_separator: "\\t"
seq_separator: "\\t"
USER_ID_FIELD: user
ITEM_ID_FIELD: item
# TIME_FIELD: time
load_col:
  inter: [user, item]
#   item: [item, director]
#   item: [item, year, title, genre, writer, director]

# model config
reg_weight: 550
embedding_size: 10 # (int) The embedding size of features.
mlp_hidden_size: [64, 32, 16] # (list of int) The hidden size of MLP layers.
dropout_prob: 0.2 # (float) The dropout rate.

# Training and evaluation config
eval_setting: RO_RS, full
epochs: 100
seed: 42
train_batch_size: 512
eval_batch_size: 512
eval_args:
  split: { "RS": [0.96, 0.02, 0.02] }
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
yaml = yaml_path
with open(yaml, "w") as f:
    f.write(cfg_str)

inter_df = pd.read_csv(os.path.join(
    data_path, 'interactions.csv'))  # 전체 학습 데이터

inter_cols = {'user': 'user:token',
              'item': 'item:token', 'time': 'time:float'}
atom_inter = convert_to_atomic(inter_df, inter_cols)
save_atomic(atom_inter, atomic_path, 'inter')

if context:
    merged_df = genre_df.groupby('item')['genre'].apply('\t'.join).reset_index()
    dfs = [writer_df, director_df, year_df, title_df]
    for df in dfs:
        if all(df.columns == writer_df.columns):
            df = df.groupby('item')['writer'].apply('\t'.join).reset_index()
        elif all(df.columns == director_df.columns):
            df = df.groupby('item')['director'].apply('\t'.join).reset_index()
        merged_df = pd.merge(merged_df, df, on='item', how='outer')

    cols = ['genre', 'writer', 'director', 'year', 'title']
    merged_df[cols] = merged_df[cols].astype(str)
    merged_df['year'] = merged_df['year'].astype(float)

    item_cols = {
        'item': 'item:token',
        'genre': 'genre:token_seq',
        'year': 'year:float',
        'writer': 'writer:token_seq',
        'director': 'director:token_seq',
        'title': 'title:token',
    }
    atom_merged = convert_to_atomic(merged_df, item_cols)
    save_atomic(atom_merged, atomic_path, 'item')

# run
run_recbole(
    model=MODEL,
    dataset=base,
    config_file_list=[yaml],
)
