import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import pyplot as plt
from recbole.quick_start import load_data_and_model
import argparse
from sklearn.cluster import KMeans


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default='saved/model.pth', help='name of models')
    # python run_inference.py --model_path=/opt/ml/input/RecBole/saved/SASRecF-Apr-07-2022_03-17-16.pth 로 실행
    
    args, _ = parser.parse_known_args()
    
    # model, dataset 불러오기
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(args.model_path)
    device = config.final_config_dict['device']
    
    user=np.array([])
    for data in dataset:
        user = np.append(user, data[0].to(device)['user_id'].cpu().numpy())
    
    user_embedding = model.interaction_matrix[user, :]
    
    num_clusters = 4
    Kmean = KMeans(n_clusters = num_clusters, random_state=42)
    ret=Kmean.fit_predict(user_embedding)
    
    tmp=[[] for _ in range(num_clusters)]
    for idx,label in enumerate(ret):
        tmp[label].append(idx)
        
    tmp_df=pd.DataFrame({'idx':[i for i in range(num_clusters)],'label':tmp})
    tmp_df.to_csv('tmp.csv',encoding='utf-8') # clustering한 뒤 label을 tmp.csv에 저장
    
    model = TSNE(n_components = 2)
    embedded = model.fit_transform(user_embedding) # user_embedding을 2차원으로 축소
    