# I-wanna-be-AI

## Dacon AI Competition 2023: Satellite Image Building Area Segmentation

- 프로젝트 개요: 위성 이미지 건물 영역 분할
- 프로젝트 기간: 2023.07.03 ~ 2023.08.17
- 프로젝트 이름: I-wanna-be-AI
- 프로젝트 목표: 위성 이미지의 건물 영역 분할(Image Segmentation)을 수행하는 AI모델을 개발

## Satellite Image Building Area Segmentation

### Dataset Info.

- train_img: TRAIN_0000.png ~ TRAIN_7139.png
- test_img: TEST_00000.png ~ TEST_60639.png

### Models

- 나중에 추가

### Results

- 나중에 수정
  - 아래 코드는 임시..

0. 0th dataset

| Model         | Accuracy | F1-score |
| ------------- | -------- | -------- |
| Random Forest | 0.822    | 0.859    |
| LightGBM      | 0.934    | 0.927    |

1. 1st dataset

| Model         | Accuracy | F1-score |
| ------------- | -------- | -------- |
| Random Forest | 0.895    | 0.897    |
| LightGBM      | 0.910    | 0.905    |

2. 2nd dataset

| Model                       | Accuracy | F1-score |
| --------------------------- | -------- | -------- |
| Random Forest               | 0.929    | 0.909    |
| LightGBM                    | 0.877    | 0.886    |
| Random Forest (Grid search) | 0.940    | 0.911    |
| LightGBM (Grid search)      | 0.941    | 0.914    |

- Hyperparameter of Random Forest: {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
- Hyperparameter of LightGBM: {'learning_rate': 0.05, 'max_depth': 5, 'min_child_samples': 250, 'n_estimators': 100, 'num_leaves': 63}

3. 3rd dataset (Final)

| Model                       | Accuracy | F1-score |
| --------------------------- | -------- | -------- |
| Random Forest               | 0.923    | 0.906    |
| LightGBM                    | 0.922    | 0.913    |
| Random Forest (Grid search) | 0.940    | 0.911    |
| LightGBM (Grid search)      | 0.941    | 0.912    |

- Hyperparameter of Random Forest: {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
- Hyperparameter of LightGBM: {'learning_rate': 0.05, 'max_depth': 2, 'min_child_samples': 250, 'n_estimators': 100, 'num_leaves': 31}

## Submissions

- img_id: 추론 위성 이미지 샘플 ID
- mask_rle : RLE 인코딩된 예측 이진마스크(0: 배경, 1 : 건물) 정보
  - 단, 예측 결과에 건물이 없는 경우 반드시 -1 처리

## Discussion

- 나중에 수정

### 3rd dataset에서 grid search 유무에 따라 feature importance에 변동 발생

1. Feature importances without grid search
2. Feature importances with grid search

- 모델이 예측할 때 사용한 feature의 조합이 변동됨을 알 수 있다.

### 결론

- 나중에 수정
- 특정 feature가 모델의 성능에 결정적인 영향을 미치는 것은 아니다.
- 그러나, 여러 feature의 조합을 이용하면 충분히 유효한 성능이 나온다.
- 따라서, 항공기 출발 시간 지연을 예측할 때 예상 출발시간, 예상 승객 수, 기상 정보 등을 복합적으로 고려해야 높은 정확도를 얻을 수 있다.

## Improvements

- 나중에 수정

1. 성능에 유의미한 변동을 줄 수 있는 feature 탐색 → 해외 항공사의 기상 혹은 해외 공항의 혼잡도를 복합적으로 사용
2. 최신 dataset을 test data로 활용 → 인천국제공항에 다시 메일 보내서 요청하기
3. LightGBM에서 feature가 추가될수록 default model의 성능이 내려가는 이유 분석
