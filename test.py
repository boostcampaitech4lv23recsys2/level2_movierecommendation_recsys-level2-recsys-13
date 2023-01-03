import pandas as pd
import torch
import numpy as np 
import argparse

from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--saved_dir", default='/opt/ml/input/base/saved/', type=str)
    parser.add_argument(
        "--checkpoint", default='GRU4Rec-Jan-01-2023_21-41-12', type=str)
    parser.add_argument("--topk", default=10, type=int)
    args = parser.parse_args()

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.saved_dir + args.checkpoint + '.pth',
    )

    external_user_ids = dataset.id2token(dataset.uid_field, list(range(dataset.user_num)))[1:]
    user_id2token = dataset.field2id_token['user']
    user_list = user_id2token[1:]

    # get top k
    from recbole.utils.case_study import full_sort_topk
    from tqdm.notebook import tqdm
    topk_items = []
    for internal_user_id in tqdm(list(range(dataset.user_num))[1:]):
        _, topk_iid_list = full_sort_topk([internal_user_id], model, test_data, k=args.topk, device=config['device'])
        external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu()).tolist()
        topk_items.extend(external_item_list)
        
    # to csv
    pd.DataFrame({'user': user_list, 'item': topk_items})\
        .explode('item')\
        .to_csv('output/' + args.checkpoint + '.csv', index=False)


if __name__ == "__main__":
    main()
