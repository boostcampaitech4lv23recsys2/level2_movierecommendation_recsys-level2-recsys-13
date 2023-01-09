# Movie Recommendation

## Project Overview
- 대회 내용 : Movie Recommendataion(사용자의 영화 시정 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측)
- 평가지표 : Recall@10

## Dataset
- Ml_item2attributes.json : 영화(item)에 대한 속성을 담은 json 파일
- directors.tsv : 영화(item)에 대한 감독 데이터
- genres.tsv : 영화(item)에 대한 장르 데이터
- titles.tsv : 영화(item)에 대한 제목 데이터
- writers.tsv : 영화(item)에 대한 작가 데이터
- years.tsv : 영화(item)에 대한 연도 데이터
- train_ratings : 31,360명의 사용자(user)가 6,807개의 영화(item)를 시청한 데이터

## Process
- EDA
  - 사용자의 시청기록 분석
  - 사용자가 시청한 영화 장르에 따른 clustering
  - 이상치 제거
- 성능 개선 실험
  - 결측치 제거 후 0.082->0.081
  - feature 추가/제거 실험
    - 새로운 feature를 생성해 실험(ratings, popularity)
- 모델
  - General model : EASE, ADMMSLIM, RecVAE, NCEPLRec
  - Context-aware model : xDeepFM, FM, FFM
  - Sequential model : S3Rec, SASRec
- 하이퍼 파라미터 튜닝
  - HyperOpt, Ray, WandB
- Ensemble
  - EASE, NCEPLRec, RecVAE, S3Rec 총 4개의 모델에 서로 다른 가중치를 부여
  ![image](https://user-images.githubusercontent.com/64139953/211254975-3df95286-a9f6-4a3d-bfb7-6d5f2858bc7c.png)


## Result
- Public Recall@10 : 0.1612
- Private Recall@10 : 0.1609
- 최종 등수 : 11/14팀
![image](https://user-images.githubusercontent.com/64139953/211251916-81646887-3c9b-4e19-bde1-b19bd893d558.png)
