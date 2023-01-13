## Feature Engineering

### 1. 추가한 feature
- ratings : 각 item에 대해 시청된 횟수가 55 이하면 0, 아니면 1
- popularity : 

### 2. 각 feature 추가/제거 실험
- 기준 모델: FM
- Data split: [0.98, 0.01, 0.01]
- Feature set: year, title, director, writer, genre
- Epoch 20

|추가한 Features|Recall@10|MRR@10|NDCG@10|Hit@10|Precision@10|
|:------:|:---:|:---:|:---:|:---:|:---:|
|안 넣은 것|0.1395|0.1269|0.0965|0.2876|0.0358|
|year|0.1377|0.1232|0.0938|0.2857|0.0348|
|title|0.1351|0.1202|0.092|0.2806|0.0343|
|director|0.1451|0.1246|0.0975|0.2889|0.0354|
|writer|0.1441|0.1230|0.0959|0.2867|0.0351|
|genre|0.1369|0.1208|0.0925|0.2836|0.0348|
|ratings|0.1464|0.0819|0.0862|0.194|0.0208|
|popularity|0.1457|0.0808|0.0855|0.1928|0.0207|
|year, title, director, writer, genre|0.1365|0.1233|0.093|0.2829|0.0346|

