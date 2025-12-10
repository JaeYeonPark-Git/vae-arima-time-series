# ARIMA 모델의 충격 완화 접근법 (Shock Relaxed Approaches With Refactored ARIMA Model)

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14.0-yellowgreen?logo=python&logoColor=yellow)](https://www.statsmodels.org/stable/index.html)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-F7931E?logo=scikit-learn)](https://scikit-learn.org/stable/)
[![Pmdarima](https://img.shields.io/badge/Pmdarima-2.0.4-8A2BE2)](https://alkaline-ml.com/pmdarima/)

**Course Project: Time Series Analysis (UNIST, Fall 2023)**

**Authors:**
* Daehyuk Bu
* **Jae Yeon Park**
* Song Kim
* Taehoon Kim

---
## 🎯 프로젝트 동기 (Motivation)

금융 시장은 복잡하며 다양한 거시적, 미시적 요인으로 인해 예측이 어렵습니다. 특히, 경제 위기나 팬데믹 같은 갑작스러운 '충격(Shock)'은 주가 데이터에 비정상적인 급등락을 발생시켜 예측을 불가능하게 만듭니다.

전통적인 시계열 모델인 **ARIMA**는 이러한 충격(이상치)에 매우 민감하게 반응하여 예측의 안정성이 떨어지는 한계가 있습니다.

본 프로젝트의 목표는 **LSTM 기반의 VAE(Variational Autoencoder)** 딥러닝 모델을 활용해 이러한 충격을 식별하고, 통계적 기법으로 데이터를 '평활화(Smoothing)'함으로써 **ARIMA 모델의 예측 안정성을 보완하고 강화**하는 것입니다.

## 💡 방법론 (Methodology)

본 프로젝트는 데이터 전처리부터 예측까지 총 3단계의 접근법을 사용합니다.

### Step 1: LSTM-VAE를 이용한 이상치(Shock) 탐지
시계열 데이터의 시간적 의존성을 포착하기 위해 LSTM 레이어를 VAE의 Encoder와 Decoder에 적용하였습니다.

* **데이터 전처리**: 비정상(Non-stationary) 시계열을 차분(Differencing)하여 정상성을 확보하고, MinMaxScaler로 스케일링을 수행했습니다.
* **모델 구조**:
    * **Encoder**: 7일치 시퀀스(Seq Size=7)를 입력받아 LSTM을 통해 잠재 공간(Latent Space, dim=8)으로 압축합니다.
    * **Sampling**: Reparameterization Trick을 사용하여 잠재 변수 $z$를 생성합니다.
    * **Decoder**: 잠재 변수로부터 원본 시퀀스의 평균(Mean)과 분산(Variance)을 복원합니다.
* **복원 오차(Reconstruction Error)**:
    * Sliding Window 방식으로 인해 겹치는 예측 구간은 **Diagonal Averaging(대각 평균화)** 기법을 사용하여 단일 시계열로 변환했습니다.
    * 원본 데이터와 복원된 데이터 간의 차이를 계산하여, 이 오차가 **상위 1% (99th percentile)를 초과**하는 지점을 '이상치(Anomaly)'로 정의합니다.

### Step 2: DBSCAN을 이용한 교차 검증
VAE가 탐지한 이상치들의 신뢰도를 높이기 위해, 밀도 기반 클러스터링 알고리즘인 **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**을 사용하여 교차 검증을 수행했습니다.
* `eps=0.15`, `min_samples=3` 파라미터를 적용하여 밀도가 낮은 지역에 위치한 데이터 포인트를 이상치로 식별했습니다.
* 두 방법론(VAE, DBSCAN) 간에 탐지된 이상치 지점이 시각적으로 **상당 부분 겹치는 것**을 확인하여 VAE 모델의 유효성을 확보했습니다.

### Step 3: 이상치 평활화 (Smoothing) 및 ARIMA 리팩토링
탐지된 이상치를 단순 제거하는 것이 아니라, VAE가 학습한 확률 분포를 활용해 값을 보정합니다.

* **Smoothing Logic**: 이상치로 판별된 시점의 실제 주가(True Close)를 VAE가 예측한 분포 범위 내로 조정합니다.
    * $$Smooth\_Close = Reconstructed\_Mean \pm (3 \times Reconstructed\_Sigma)$$
* **Refactored ARIMA**: 평활화된(Smoothed) 데이터를 사용하여 **Auto-ARIMA**($p, d, q$ 최적화)를 수행하고, 이를 기반으로 미래 주가를 예측합니다.

![VAE Smoothing Process](./assets/vae_smoothing_graph.png)

*(리포트 Figure 6: VAE를 통한 이상치 평활화 예시)*

## 📊 실험 환경 및 결과 (Configuration & Results)

### 실험 환경 (Configuration)
코드(`analysis.py`)에서 사용된 주요 파라미터는 다음과 같습니다.
* **Target Asset**: Coca-Cola (KO)
* **Data Period**: 2000-01-01 ~ 2020-12-31
* **Window Size**: 7 days
* **Latent Dimension**: 8
* **Smoothing Multiplier**: $3\sigma$

### 실험 결과
'평활화'를 적용한 모델(Ref 99)이 순수 ARIMA 모델(Naïve Model)보다 모든 평가지표(MAE, MSE, RMSE)에서 더 우수한 예측 성능을 보였습니다.

| Model | MAE | MSE | RMSE | MPE |
| :--- | :---: | :---: | :---: | :---: |
| Naïve Model | 0.317 | 0.144 | 0.379 | 80.044 |
| Ref 95 | 0.346 | 0.238 | 0.488 | 74.432 |
| **Ref 99 (Best)** | **0.287** | **0.131** | **0.362** | **74.482** |
*(리포트 Table 3: 모델별 예측 성능 비교)*

![Refactored ARIMA Prediction](./assets/refactored_arima_graph.png)

*(리포트 Figure 7: 최종 예측 비교. Re-factored 99 ARIMA(초록색)가 기존 ARIMA(빨간색)보다 실제 값(파란색)에 더 근접함)*

## 🏁 결론 및 한계

**결론:** VAE를 통해 시계열 데이터의 충격(이상치)을 탐지하고, 이를 통계적으로 유의미한 값으로 평활화(smoothing)하는 접근법은 기존 ARIMA 모델의 안정성과 예측 정확도를 향상시킬 수 있음을 확인했습니다.

**한계:** 이 '평활화' 과정은 ARIMA 모델의 한계(비선형성, 이상치 민감성)를 보완하기 위한 일종의 데이터 전처리이며, 원본 데이터를 수정한다는 점에서 '데이터 조작(data manipulation)'으로 해석될 여지가 있습니다.
