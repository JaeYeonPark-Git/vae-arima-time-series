# --- 1. 모듈 임포트 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import FinanceDataReader as fdr
import math

# Statsmodels and ARIMA
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima

# Scikit-learn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.cluster import DBSCAN

# Keras (TensorFlow)
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, LSTM, RepeatVector, TimeDistributed, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback

print("All modules imported.")

# --- 2. 데이터 로드 및 전처리 ---
print("Loading data...")
start_date = '2000-01-01'
end_date = '2020-12-31'
raw_df = fdr.DataReader('KO', start_date, end_date) # Coca-Cola (KO) [cite: 160, 322]
df = raw_df[['Close']].copy()

# 차분 및 스케일링
diff_df = df.diff().fillna(method='bfill')
train = diff_df.copy()
train = train.reset_index()

scaler = MinMaxScaler().fit(train[['Close']])
train['Close'] = scaler.transform(train[['Close']])

# 시퀀스 데이터 생성
seq_size = 7 # 7-day time steps [cite: 66]
def to_sequence(x, y, seq_size=1):
    x_values = []
    y_values = []
    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])
    return np.array(x_values), np.array(y_values)

X_train, Y_train_seq = to_sequence(train[['Close']], train['Close'], seq_size)
print(f"X_train shape: {X_train.shape}")

# --- 3. VAE 모델링 ---
print("Building VAE model...")
X_values = X_train.copy()
pi = K.constant(math.pi)
input_dim = X_values.shape[2] # 1
time_steps = X_values.shape[1] # 7
intermediate_dim = 64
latent_dim = 8

# 3-1. Encoder
inputs = Input(shape=(time_steps, input_dim))
[cite_start]h = LSTM(intermediate_dim, activation='relu', recurrent_activation='sigmoid', recurrent_dropout=0.4, unroll=True)(inputs) [cite: 68]
h = BatchNormalization()(h)
h = Dropout(0.4)(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# 3-2. Latent Space (Sampling)
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var) * epsilon # Reparameterization Trick

z = Lambda(sampling)([z_mean, z_log_var])

# 3-3. Decoder
repeat_layer = RepeatVector(time_steps)
decoder_h = LSTM(intermediate_dim, activation='relu', recurrent_activation='sigmoid', return_sequences=True, recurrent_dropout=0.4, unroll=True)
decoder_mean = TimeDistributed(Dense(input_dim, 'sigmoid'))
decoder_var = TimeDistributed(Dense(input_dim, activation='softplus')) # [cite: 393]

h_decoded = repeat_layer(z)
h_decoded = decoder_h(h_decoded)
h_decoded = BatchNormalization()(h_decoded)
x_decoded_mean = decoder_mean(h_decoded)
x_decoded_var = decoder_var(h_decoded)

# 3-4. VAE 모델 컴파일
prob_model = Model(inputs, [z_mean, z_log_var, x_decoded_mean, x_decoded_var, h_decoded])

# Loss
reconstruction_loss = 0.5 * (K.square(inputs - x_decoded_mean) / (x_decoded_var + K.epsilon()) + K.log(x_decoded_var + K.epsilon()) + K.log(2 * pi))
reconstruction_loss = K.sum(reconstruction_loss, axis=[1, 2])
reconstruction_loss /= input_dim

kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

vae_loss = K.mean(reconstruction_loss + kl_loss)

opt = optimizers.Adam(learning_rate=5e-5)
prob_model.add_loss(vae_loss)
prob_model.compile(optimizer=opt)

prob_model.summary()

# (모델 훈련)
# prob_model.fit(X_values, epochs=N, batch_size=N) # 실제 훈련 코드 (보고서엔 생략됨)
# 여기서는 훈련 대신 예측을 바로 진행 (저장된 모델이 있다고 가정하거나, 예측만 수행)
print("Predicting with VAE model...")
z_mean, z_log_var, x_decoded_mean, x_decoded_var, h_decoded = prob_model.predict(X_values)

# --- 4. 이상치 탐지 및 평활화 (Anomaly Detection & Smoothing) ---
print("Detecting anomalies...")

# VAE 예측값 후처리 (Diagonal Averaging)
# (보고서의 diagonal_avg 함수는 VAE의 7일치 예측을 단일 예측값으로 변환)
def diagonal_avg(df):
    days = df[['day', 'day+1', 'day+2', 'day+3', 'day+4', 'day+5', 'day+6']]
    head_x = 0
    head_y = 0
    diag_ave = []
    while head_y < 7: # 7 is the number of columns
        sum_val = 0
        x, y = head_x, head_y
        count = 0
        while y >= 0:
            sum_val += days.iloc[x, y]
            x += 1
            y -= 1
            count += 1
        sum_val = sum_val / count if count > 0 else 0
        diag_ave.append(sum_val)
        head_y += 1
    
    head_x = 1
    head_y = 6
    while head_x < len(df): # number of rows
        sum_val = 0
        x, y = head_x, head_y
        count = 0
        while y >= 0 and x < len(df):
            sum_val += days.iloc[x, y]
            x += 1
            y -= 1
            count += 1
        sum_val = sum_val / count if count > 0 else 0
        diag_ave.append(sum_val)
        head_x += 1
        
    df_diag_ave = pd.DataFrame(diag_ave, columns=['day'])
    return df_diag_ave

# 예측값(mean)을 7일 윈도우에서 일별 값으로 변환
temp_mean = []
for i in range(len(x_decoded_mean)):
    temp_mean.append(x_decoded_mean[i].reshape(-1))
temp_mean = pd.DataFrame(temp_mean, columns=['day', 'day+1', 'day+2', 'day+3', 'day+4', 'day+5', 'day+6'])
temp_mean['Date'] = train['Date'].iloc[:len(temp_mean)]
temp_mean = temp_mean.set_index('Date')
df_diag_ave = diagonal_avg(temp_mean.reset_index())
df_diag_ave.index = df[1:].index

# 예측값(var)도 동일하게 변환
temp_var = []
for i in range(len(x_decoded_var)):
    temp_var.append(x_decoded_var[i].reshape(-1))
temp_var = pd.DataFrame(temp_var, columns=['day', 'day+1', 'day+2', 'day+3', 'day+4', 'day+5', 'day+6'])
temp_var['Date'] = train['Date'].iloc[:len(temp_var)]
temp_var = temp_var.set_index('Date')
df_diag_var = diagonal_avg(temp_var.reset_index())
df_diag_var.index = df[1:].index

# Y_train (실제값) 과 pred_df (예측값) 정렬
Y_train = train[1:].set_index('Date').copy()
Y_train = Y_train.iloc[:-seq_size]
pred_df = df_diag_ave.iloc[:-seq_size]
pred_var_df = df_diag_var.iloc[:-seq_size]

# [cite_start]복원 오차 계산 [cite: 94]
temp_df = pd.concat([Y_train['Close'], pred_df['day']], axis=1)
temp_df['cal'] = 0
for i in temp_df.index:
    if temp_df.loc[i, 'Close'] >= temp_df.loc[i, 'day']:
        temp_df.loc[i, 'cal'] = temp_df.loc[i, 'Close'] - temp_df.loc[i, 'day']
    else:
        temp_df.loc[i, 'cal'] = temp_df.loc[i, 'day'] - temp_df.loc[i, 'Close']
reconstruction_error = temp_df['cal']

# 99 백분위수(Percentile)를 임계값으로 설정 [cite: 95]
threshold = np.percentile(reconstruction_error, 99)
outliers = np.where(reconstruction_error > threshold)
print("Anomaly Indices:", outliers)

outlier_idx = np.zeros(pred_df.shape[0], dtype=int)
indices = outliers[0]
outlier_idx[indices] = 1

# 평활화(Smoothing) 데이터프레임 생성
ano_df = pd.DataFrame(pred_df.copy())
ano_df['anomal'] = outlier_idx
ano_df['sigma'] = pred_var_df['day'].values
ano_df['sigma'] = np.sqrt(ano_df['sigma']) # 분산 -> 표준편차

# 시그마 값 클리핑
sig_scaler = MinMaxScaler()
ano_df['sigma'] = sig_scaler.fit_transform(ano_df[['sigma']])
ano_sig_min = ano_df['sigma'].quantile(0.05)
ano_sig_max = ano_df['sigma'].quantile(0.95)
ano_df.loc[ano_df['sigma'] > ano_sig_max, 'sigma'] = ano_sig_max
ano_df.loc[ano_df['sigma'] < ano_sig_min, 'sigma'] = ano_sig_min # (코드 오타 수정: ano_sig_max -> ano_sig_min)

ano_df['true_close'] = Y_train['Close']
ano_df['smooth_close'] = ano_df['true_close'].copy()

# 평활화(Smoothing) 적용 [cite: 196, 197]
print("Applying smoothing to anomalies...")
for idx in ano_df.index:
    if ano_df.loc[idx, 'anomal'] == 1:
        if ano_df.loc[idx, 'day'] < ano_df.loc[idx, 'true_close']:
            # (reconstructed mean + 3*sigma)
            ano_df.loc[idx, 'smooth_close'] = ano_df.loc[idx, 'day'] + 3 * ano_df.loc[idx, 'sigma']
        else:
            # (reconstructed mean - 3*sigma)
            ano_df.loc[idx, 'smooth_close'] = ano_df.loc[idx, 'day'] - 3 * ano_df.loc[idx, 'sigma']

# 원래 스케일로 복원
final_df = pd.DataFrame([], columns=['true_close', 'smooth_close'])
final_df['true_close'] = scaler.inverse_transform(ano_df[['true_close']]).reshape(-1)
final_df['smooth_close'] = scaler.inverse_transform(ano_df[['smooth_close']]).reshape(-1)
final_df['anomal'] = ano_df['anomal'].values
final_df.index = Y_train.index

# --- 5. 최종 ARIMA 모델 테스트 ---
print("Running final ARIMA vs. Refactored ARIMA test...")
# Naive ARIMA (원본 데이터 사용)
X_naive = final_df.iloc[:-7, 0] # true_close
y_naive = final_df.iloc[-7:, 0] # true_close
naive_model = auto_arima(X_naive, max_p=10, max_q=10, max_d=3, alpha=0.05, n_jobs=1, seasonal=False)
pred_naive = naive_model.predict(n_periods=7, return_conf_int=True)
y_pred_naive = pred_naive[0]
mse_naive = mean_squared_error(y_pred_naive, y_naive)
print(f"Naive ARIMA MSE: {mse_naive}")

# Refactored ARIMA (평활화된 데이터 사용)
X_refactored = final_df.iloc[:-7, 1] # smooth_close
y_refactored = final_df.iloc[-7:, 0] # true_close (타겟은 동일)
refactored_model = auto_arima(X_refactored, max_p=10, max_q=10, max_d=3, alpha=0.05, n_jobs=1, seasonal=False)
pred_refactored = refactored_model.predict(n_periods=7, return_conf_int=True)
y_pred_refactored = pred_refactored[0]
mse_refactored = mean_squared_error(y_pred_refactored, y_naive) # 타겟은 y_naive (true_close)
print(f"Refactored ARIMA MSE: {mse_refactored}")

# [cite_start]시각화 (Figure 7) [cite: 263]
plot_final_df = final_df.iloc[-30:, 0].copy() # true_close
pred_idx = y_naive.index

plt.figure(figsize=(12, 6))
plt.plot(plot_final_df.index, plot_final_df.values, label='Close')
plt.vlines(pred_idx[0], plot_final_df.min(), plot_final_df.max(), linestyles='dashed', color='black', label='Forecast')
plt.plot(pred_idx, y_pred_naive, label='forcasting_ARIMA', color='r')
plt.plot(pred_idx, y_pred_refactored, label='forcasting_Re-factored 99 ARIMA', color='g') # 초록색으로 변경
# plt.fill_between(pred_idx, pred_lb, pred_ub, color='k', alpha=0.1, label='95% Confidence Interval') # 원본 pred 사용
plt.legend(loc='upper left')
plt.title("ARIMA vs. Re-factored ARIMA (Ref 99)")
plt.show()

# --- 6. DBSCAN 교차 검증 ---
print("Running DBSCAN validation...")
# 스케일링
array_idx_diff_df = np.array([i / (len(diff_df) - 1) for i in range(len(diff_df))])
mm_scaler_db = MinMaxScaler()
diff_df['time_idx'] = array_idx_diff_df
diff_df['scaled_close'] = mm_scaler_db.fit_transform(diff_df[['Close']])

# DBSCAN 모델
eps_diff_df = 0.15
[cite_start]model_for_diff_df = DBSCAN(eps=eps_diff_df, min_samples=3) [cite: 113, 615]
diff_df['cluster'] = model_for_diff_df.fit_predict(diff_df[['time_idx', 'scaled_close']])

outlier_count = len(diff_df[diff_df['cluster'] == -1])
print(f"DBSCAN detected outliers: {outlier_count}")
# print(diff_df[diff_df['cluster'] == -1])

print("Script finished.")
