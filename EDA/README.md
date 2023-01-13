## EDA

### 1. user 시청 기록 분석/이상치 제거
<p align="center">
<img width="300" alt="Screenshot 2023-01-06 at 10 19 32 AM" src="https://user-images.githubusercontent.com/64139953/212221991-e4d682bf-f320-40b6-b3a9-9a28045ad647.png">  

- 타이틀은 동일하나 item id가 다른 데이터가 존재함.
- 데이터를 확인 결과 해당 연도에 같은 이름을 가진 영화가 하나만 존재한다는 사실에 둘 중 하나가 잘못된 데이터라고 판단

➜ train_ratings 데이터를 확인해본 결과 34048 id의 데이터가 64997 id 데이터보다 16배 많이 존재해 64997 id를 34048로 대체함.

### 2. user clustering
2.1 **user가 시청한 영화 장르에 따른 clustering**
  - 각 user가 시청한 item에 대한 genre 정보를 multi-hot encoding으로 표현함.
  - 이후 모든 item의 vector를 합산한 후 정규화를 진행함.
  - 총 4개의 cluster로 분류함.
<p align="center">
<img width="300" alt="Screenshot 2023-01-05 at 1 45 34 PM" src="https://user-images.githubusercontent.com/64139953/212223242-e0209aaf-4727-4ae1-93b4-0a42f2f040c8.png">

2.2 **EASE 모델의 user embedding 값을 이용한 clustering**
  - user embedding 값을 추출해 총 4개의 cluster로 분류함.
<p align="center">
<img width="300" src="https://user-images.githubusercontent.com/64139953/212223630-c2357206-6c8f-4351-a258-921c074ace1c.png">
