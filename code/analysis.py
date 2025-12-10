"""
Stock Price Anomaly Detection and Forecasting using VAE and ARIMA

This script implements:
1. Variational Autoencoder (VAE) for anomaly detection
2. ARIMA models for time series forecasting
3. DBSCAN clustering for outlier validation

"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
import math

# Statsmodels and ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA  # Updated import
except ImportError:
    from statsmodels.tsa.arima_model import ARIMA  # Fallback for older versions

from pmdarima.arima import auto_arima

# Scikit-learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import DBSCAN

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Lambda, Dropout, LSTM, RepeatVector, 
    TimeDistributed, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

# Configuration
CONFIG = {
    'symbol': 'KO',  # Coca-Cola
    'start_date': '2000-01-01',
    'end_date': '2020-12-31',
    'seq_size': 7,
    'intermediate_dim': 64,
    'latent_dim': 8,
    'learning_rate': 5e-5,
    'anomaly_percentile': 99,
    'sigma_multiplier': 3,
    'arima_max_p': 10,
    'arima_max_q': 10,
    'arima_max_d': 3,
    'dbscan_eps': 0.15,
    'dbscan_min_samples': 3,
    'output_dir': './results',
    'plot_last_n_days': 30,
    'forecast_periods': 7
}


def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_stock_data(symbol, start_date, end_date):
    """
    Load stock data from FinanceDataReader.
    
    Args:
        symbol: Stock symbol (e.g., 'KO')
        start_date: Start date string
        end_date: End date string
    
    Returns:
        DataFrame with stock data
    """
    try:
        raw_df = fdr.DataReader(symbol, start_date, end_date)
        if raw_df.empty:
            raise ValueError(f"No data retrieved for symbol {symbol}")
        
        df = raw_df[['Close']].copy()
        if df.empty:
            raise ValueError("Close price data is empty")
        
        print(f"Data loaded successfully: {len(df)} rows")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {str(e)}")


def preprocess_data(df):
    """
    Preprocess stock data: difference, fill NaN, and scale.
    
    Args:
        df: DataFrame with 'Close' column
    
    Returns:
        Tuple of (differenced_df, scaled_train_df, scaler)
    """
    # Difference
    diff_df = df.diff()
    
    # Fill NaN - use bfill() method instead of deprecated fillna(method='bfill')
    diff_df = diff_df.bfill()
    
    # Prepare training data
    train = diff_df.copy().reset_index()
    
    # Ensure Date column exists
    if 'Date' not in train.columns and train.index.name != 'Date':
        if 'index' in train.columns:
            train = train.rename(columns={'index': 'Date'})
        elif train.index.name is None:
            train.index.name = 'Date'
            train = train.reset_index()
    
    # Scale
    scaler = MinMaxScaler()
    train['Close'] = scaler.fit_transform(train[['Close']])
    
    return diff_df, train, scaler


def to_sequence(x, y, seq_size=1):
    """
    Convert time series data to sequences.
    
    Args:
        x: Input features DataFrame
        y: Target Series
        seq_size: Sequence length
    
    Returns:
        Tuple of (X sequences, y sequences)
    """
    x_values = []
    y_values = []
    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])
    return np.array(x_values), np.array(y_values)


def build_vae_model(time_steps, input_dim, intermediate_dim, latent_dim, learning_rate):
    """
    Build Variational Autoencoder model.
    
    Args:
        time_steps: Number of time steps in sequence
        input_dim: Input dimension
        intermediate_dim: Intermediate layer dimension
        latent_dim: Latent space dimension
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled VAE model
    """
    # Encoder
    inputs = Input(shape=(time_steps, input_dim))
    h = LSTM(intermediate_dim, activation='relu', recurrent_activation='sigmoid', 
             recurrent_dropout=0.4, unroll=True)(inputs)
    h = BatchNormalization()(h)
    h = Dropout(0.4)(h)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    
    # Sampling
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon  # Fixed: should divide by 2
    
    z = Lambda(sampling)([z_mean, z_log_var])
    
    # Decoder
    repeat_layer = RepeatVector(time_steps)
    decoder_h = LSTM(intermediate_dim, activation='relu', recurrent_activation='sigmoid', 
                     return_sequences=True, recurrent_dropout=0.4, unroll=True)
    decoder_mean = TimeDistributed(Dense(input_dim, activation='sigmoid'))
    decoder_var = TimeDistributed(Dense(input_dim, activation='softplus'))
    
    h_decoded = repeat_layer(z)
    h_decoded = decoder_h(h_decoded)
    h_decoded = BatchNormalization()(h_decoded)
    x_decoded_mean = decoder_mean(h_decoded)
    x_decoded_var = decoder_var(h_decoded)
    
    # Model
    prob_model = Model(inputs, [z_mean, z_log_var, x_decoded_mean, x_decoded_var, h_decoded])
    
    # Loss
    pi = K.constant(math.pi)
    reconstruction_loss = 0.5 * (
        K.square(inputs - x_decoded_mean) / (x_decoded_var + K.epsilon()) + 
        K.log(x_decoded_var + K.epsilon()) + 
        K.log(2 * pi)
    )
    reconstruction_loss = K.sum(reconstruction_loss, axis=[1, 2])
    reconstruction_loss /= input_dim
    
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    
    opt = optimizers.Adam(learning_rate=learning_rate)
    prob_model.add_loss(vae_loss)
    prob_model.compile(optimizer=opt)
    
    return prob_model


def diagonal_avg(df):
    """
    Apply diagonal averaging to convert multi-step predictions to single values.
    
    Args:
        df: DataFrame with columns ['day', 'day+1', ..., 'day+6']
    
    Returns:
        DataFrame with averaged values
    """
    day_cols = [f'day+{i}' if i > 0 else 'day' for i in range(7)]
    
    # Check if columns exist
    available_cols = [col for col in day_cols if col in df.columns]
    if not available_cols:
        raise ValueError("Required day columns not found in DataFrame")
    
    days = df[available_cols]
    diag_ave = []
    
    # First part: from top-left to top-right
    for head_y in range(len(available_cols)):
        sum_val = 0
        x, y = 0, head_y
        count = 0
        while y >= 0 and x < len(days):
            sum_val += days.iloc[x, y]
            x += 1
            y -= 1
            count += 1
        diag_ave.append(sum_val / count if count > 0 else 0)
    
    # Second part: from top-right downward
    for head_x in range(1, len(days)):
        sum_val = 0
        x, y = head_x, len(available_cols) - 1
        count = 0
        while y >= 0 and x < len(days):
            sum_val += days.iloc[x, y]
            x += 1
            y -= 1
            count += 1
        diag_ave.append(sum_val / count if count > 0 else 0)
    
    return pd.DataFrame(diag_ave, columns=['day'])


def detect_anomalies(Y_train, pred_df, percentile=99):
    """
    Detect anomalies using reconstruction error.
    
    Args:
        Y_train: Actual values DataFrame
        pred_df: Predicted values DataFrame
        percentile: Percentile threshold for anomaly detection
    
    Returns:
        Tuple of (reconstruction_error, threshold, outlier_indices)
    """
    # Calculate reconstruction error
    temp_df = pd.concat([Y_train['Close'], pred_df['day']], axis=1)
    temp_df['cal'] = np.abs(temp_df['Close'] - temp_df['day'])  # Simplified
    
    reconstruction_error = temp_df['cal']
    threshold = np.percentile(reconstruction_error, percentile)
    outliers = np.where(reconstruction_error > threshold)[0]
    
    return reconstruction_error, threshold, outliers


def apply_smoothing(ano_df, sigma_multiplier=3):
    """
    Apply smoothing to detected anomalies.
    
    Args:
        ano_df: DataFrame with anomaly flags and predictions
        sigma_multiplier: Multiplier for sigma in smoothing
    
    Returns:
        DataFrame with smoothed values
    """
    ano_df = ano_df.copy()
    ano_df['smooth_close'] = ano_df['true_close'].copy()
    
    for idx in ano_df.index:
        if ano_df.loc[idx, 'anomal'] == 1:
            if ano_df.loc[idx, 'day'] < ano_df.loc[idx, 'true_close']:
                ano_df.loc[idx, 'smooth_close'] = (
                    ano_df.loc[idx, 'day'] + sigma_multiplier * ano_df.loc[idx, 'sigma']
                )
            else:
                ano_df.loc[idx, 'smooth_close'] = (
                    ano_df.loc[idx, 'day'] - sigma_multiplier * ano_df.loc[idx, 'sigma']
                )
    
    return ano_df


def train_arima_models(final_df, forecast_periods=7, max_p=10, max_q=10, max_d=3):
    """
    Train and compare Naive ARIMA and Refactored ARIMA models.
    
    Args:
        final_df: DataFrame with true_close and smooth_close
        forecast_periods: Number of periods to forecast
        max_p, max_q, max_d: ARIMA parameters
    
    Returns:
        Dictionary with predictions and MSEs
    """
    # Naive ARIMA
    X_naive = final_df.iloc[:-forecast_periods, 0]  # true_close
    y_naive = final_df.iloc[-forecast_periods:, 0]  # true_close
    
    naive_model = auto_arima(
        X_naive, max_p=max_p, max_q=max_q, max_d=max_d, 
        alpha=0.05, n_jobs=1, seasonal=False, suppress_warnings=True
    )
    pred_naive = naive_model.predict(n_periods=forecast_periods, return_conf_int=True)
    y_pred_naive = pred_naive[0]
    mse_naive = mean_squared_error(y_pred_naive, y_naive)
    
    # Refactored ARIMA
    X_refactored = final_df.iloc[:-forecast_periods, 1]  # smooth_close
    refactored_model = auto_arima(
        X_refactored, max_p=max_p, max_q=max_q, max_d=max_d,
        alpha=0.05, n_jobs=1, seasonal=False, suppress_warnings=True
    )
    pred_refactored = refactored_model.predict(n_periods=forecast_periods, return_conf_int=True)
    y_pred_refactored = pred_refactored[0]
    mse_refactored = mean_squared_error(y_pred_refactored, y_naive)
    
    return {
        'y_naive': y_naive,
        'y_pred_naive': y_pred_naive,
        'y_pred_refactored': y_pred_refactored,
        'mse_naive': mse_naive,
        'mse_refactored': mse_refactored
    }


def plot_results(final_df, arima_results, output_dir, last_n_days=30):
    """
    Plot and save comparison results.
    
    Args:
        final_df: DataFrame with results
        arima_results: Dictionary with ARIMA predictions
        output_dir: Output directory path
        last_n_days: Number of days to plot
    """
    plot_final_df = final_df.iloc[-last_n_days:, 0].copy()
    pred_idx = arima_results['y_naive'].index
    
    plt.figure(figsize=(12, 6))
    plt.plot(plot_final_df.index, plot_final_df.values, label='Close', linewidth=2)
    plt.vlines(
        pred_idx[0], plot_final_df.min(), plot_final_df.max(), 
        linestyles='dashed', color='black', label='Forecast Start'
    )
    plt.plot(
        pred_idx, arima_results['y_pred_naive'], 
        label='ARIMA (Naive)', color='r', marker='o', markersize=4
    )
    plt.plot(
        pred_idx, arima_results['y_pred_refactored'], 
        label='ARIMA (Refactored)', color='g', marker='s', markersize=4
    )
    plt.legend(loc='upper left')
    plt.title("ARIMA vs. Re-factored ARIMA Comparison")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'arima_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()


def run_dbscan_validation(diff_df, eps=0.15, min_samples=3):
    """
    Run DBSCAN clustering for outlier validation.
    
    Args:
        diff_df: Differenced DataFrame
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter
    
    Returns:
        Number of detected outliers
    """
    # Scale
    array_idx_diff_df = np.array([i / (len(diff_df) - 1) for i in range(len(diff_df))])
    mm_scaler_db = MinMaxScaler()
    diff_df_scaled = diff_df.copy()
    diff_df_scaled['time_idx'] = array_idx_diff_df
    diff_df_scaled['scaled_close'] = mm_scaler_db.fit_transform(diff_df[['Close']])
    
    # DBSCAN
    model_for_diff_df = DBSCAN(eps=eps, min_samples=min_samples)
    diff_df_scaled['cluster'] = model_for_diff_df.fit_predict(
        diff_df_scaled[['time_idx', 'scaled_close']]
    )
    
    outlier_count = len(diff_df_scaled[diff_df_scaled['cluster'] == -1])
    return outlier_count


def main():
    """Main execution function."""
    print("=" * 60)
    print("Stock Price Anomaly Detection and Forecasting")
    print("=" * 60)
    
    # Create output directory
    output_dir = create_output_directory(CONFIG['output_dir'])
    
    try:
        # 1. Load data
        print("\n[1/6] Loading stock data...")
        df = load_stock_data(CONFIG['symbol'], CONFIG['start_date'], CONFIG['end_date'])
        
        # 2. Preprocess data
        print("\n[2/6] Preprocessing data...")
        diff_df, train, scaler = preprocess_data(df)
        
        # 3. Create sequences
        print("\n[3/6] Creating sequences...")
        X_train, Y_train_seq = to_sequence(
            train[['Close']], train['Close'], CONFIG['seq_size']
        )
        print(f"X_train shape: {X_train.shape}")
        
        # 4. Build and use VAE model
        print("\n[4/6] Building VAE model...")
        time_steps = X_train.shape[1]
        input_dim = X_train.shape[2]
        
        vae_model = build_vae_model(
            time_steps, input_dim, CONFIG['intermediate_dim'], 
            CONFIG['latent_dim'], CONFIG['learning_rate']
        )
        vae_model.summary()
        
        print("Predicting with VAE model...")
        X_values = X_train.copy()
        z_mean, z_log_var, x_decoded_mean, x_decoded_var, h_decoded = vae_model.predict(
            X_values, verbose=0
        )
        
        # 5. Process VAE predictions
        print("\n[5/6] Processing VAE predictions and detecting anomalies...")
        
        # Convert predictions to DataFrames
        temp_mean = pd.DataFrame(
            x_decoded_mean.reshape(len(x_decoded_mean), -1),
            columns=[f'day+{i}' if i > 0 else 'day' for i in range(7)]
        )
        
        # Ensure Date column exists
        if 'Date' in train.columns:
            temp_mean['Date'] = train['Date'].iloc[:len(temp_mean)].values
        else:
            temp_mean['Date'] = train.index[:len(temp_mean)]
        
        temp_mean = temp_mean.set_index('Date')
        df_diag_ave = diagonal_avg(temp_mean.reset_index())
        
        # Align indices
        if len(df_diag_ave) > len(df) - 1:
            df_diag_ave = df_diag_ave.iloc[:len(df) - 1]
        df_diag_ave.index = df.index[1:len(df_diag_ave) + 1]
        
        # Same for variance
        temp_var = pd.DataFrame(
            x_decoded_var.reshape(len(x_decoded_var), -1),
            columns=[f'day+{i}' if i > 0 else 'day' for i in range(7)]
        )
        if 'Date' in train.columns:
            temp_var['Date'] = train['Date'].iloc[:len(temp_var)].values
        else:
            temp_var['Date'] = train.index[:len(temp_var)]
        temp_var = temp_var.set_index('Date')
        df_diag_var = diagonal_avg(temp_var.reset_index())
        if len(df_diag_var) > len(df) - 1:
            df_diag_var = df_diag_var.iloc[:len(df) - 1]
        df_diag_var.index = df.index[1:len(df_diag_var) + 1]
        
        # Prepare Y_train
        Y_train = train[1:].set_index('Date').copy() if 'Date' in train.columns else train[1:].copy()
        Y_train = Y_train.iloc[:-CONFIG['seq_size']]
        
        # Align pred_df
        pred_df = df_diag_ave.iloc[:-CONFIG['seq_size']]
        pred_var_df = df_diag_var.iloc[:-CONFIG['seq_size']]
        
        # Ensure alignment
        common_idx = Y_train.index.intersection(pred_df.index)
        Y_train = Y_train.loc[common_idx]
        pred_df = pred_df.loc[common_idx]
        pred_var_df = pred_var_df.loc[common_idx]
        
        # Detect anomalies
        reconstruction_error, threshold, outliers = detect_anomalies(
            Y_train, pred_df, CONFIG['anomaly_percentile']
        )
        print(f"Anomaly threshold (99th percentile): {threshold:.6f}")
        print(f"Number of anomalies detected: {len(outliers)}")
        
        # Create anomaly DataFrame
        outlier_idx = np.zeros(len(pred_df), dtype=int)
        outlier_idx[outliers] = 1
        
        ano_df = pred_df.copy()
        ano_df['anomal'] = outlier_idx
        ano_df['sigma'] = np.sqrt(pred_var_df['day'].values)
        
        # Clip sigma values
        sig_scaler = MinMaxScaler()
        ano_df['sigma'] = sig_scaler.fit_transform(ano_df[['sigma']])
        ano_sig_min = ano_df['sigma'].quantile(0.05)
        ano_sig_max = ano_df['sigma'].quantile(0.95)
        ano_df.loc[ano_df['sigma'] > ano_sig_max, 'sigma'] = ano_sig_max
        ano_df.loc[ano_df['sigma'] < ano_sig_min, 'sigma'] = ano_sig_min
        
        ano_df['true_close'] = Y_train['Close'].values
        ano_df = apply_smoothing(ano_df, CONFIG['sigma_multiplier'])
        
        # Inverse transform
        final_df = pd.DataFrame()
        final_df['true_close'] = scaler.inverse_transform(ano_df[['true_close']]).reshape(-1)
        final_df['smooth_close'] = scaler.inverse_transform(ano_df[['smooth_close']]).reshape(-1)
        final_df['anomal'] = ano_df['anomal'].values
        final_df.index = ano_df.index
        
        # Save results
        results_path = os.path.join(output_dir, 'anomaly_detection_results.csv')
        final_df.to_csv(results_path)
        print(f"Results saved to: {results_path}")
        
        # 6. ARIMA models
        print("\n[6/6] Training ARIMA models...")
        arima_results = train_arima_models(
            final_df, CONFIG['forecast_periods'],
            CONFIG['arima_max_p'], CONFIG['arima_max_q'], CONFIG['arima_max_d']
        )
        print(f"Naive ARIMA MSE: {arima_results['mse_naive']:.6f}")
        print(f"Refactored ARIMA MSE: {arima_results['mse_refactored']:.6f}")
        
        # Plot results
        plot_results(final_df, arima_results, output_dir, CONFIG['plot_last_n_days'])
        
        # 7. DBSCAN validation
        print("\n[7/7] Running DBSCAN validation...")
        outlier_count = run_dbscan_validation(
            diff_df, CONFIG['dbscan_eps'], CONFIG['dbscan_min_samples']
        )
        print(f"DBSCAN detected outliers: {outlier_count}")
        
        print("\n" + "=" * 60)
        print("Script completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
