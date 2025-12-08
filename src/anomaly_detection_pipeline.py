import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# --- Configuration ---
LOG_FILE = 'pipeline.log'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(filename=os.path.join(OUTPUT_DIR, LOG_FILE), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Anomaly Detection Pipeline Initialized.")

# --- Phase 2: Data Generation and EDA ---

def generate_synthetic_data(n_samples=1000, n_features=3):
    """
    Generates synthetic multivariate time series data with embedded anomalies.
    Features: Temperature, Pressure, Vibration.
    """
    logging.info("Starting synthetic data generation.")
    np.random.seed(42)
    
    # 1. Base Data Generation (Normal behavior)
    time = pd.date_range(start='2025-01-01', periods=n_samples, freq='H')
    
    # Feature 1: Temperature (Sinusoidal trend + noise)
    temp_base = 25 + 5 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
    temp_noise = np.random.normal(0, 0.5, n_samples)
    temperature = temp_base + temp_noise
    
    # Feature 2: Pressure (Linear trend + noise, correlated with temp)
    pressure_base = 100 + 0.01 * np.arange(n_samples) + 0.5 * (temperature - temperature.mean())
    pressure_noise = np.random.normal(0, 0.3, n_samples)
    pressure = pressure_base + pressure_noise
    
    # Feature 3: Vibration (Random walk + noise)
    vibration_base = np.cumsum(np.random.normal(0, 0.1, n_samples))
    vibration_noise = np.random.normal(0, 0.1, n_samples)
    vibration = 5 + vibration_base + vibration_noise
    
    df = pd.DataFrame({'Temperature': temperature, 'Pressure': pressure, 'Vibration': vibration}, index=time)
    df['is_anomaly'] = 0 # Initialize anomaly label
    
    # 2. Embed Anomalies
    
    # Anomaly 1: Point Anomaly (Extreme spike in Temperature)
    df.loc[df.index[200], 'Temperature'] += 20 
    df.loc[df.index[200], 'is_anomaly'] = 1
    
    # Anomaly 2: Contextual Anomaly (High Pressure during low Temperature period)
    df.loc[df.index[450:470], 'Pressure'] += 5 
    df.loc[df.index[450:470], 'is_anomaly'] = 1
    
    # Anomaly 3: Collective Anomaly (Slight, sustained drift in Vibration)
    df.loc[df.index[700:800], 'Vibration'] += 3 
    df.loc[df.index[700:800], 'is_anomaly'] = 1
    
    logging.info(f"Generated {n_samples} samples with {df['is_anomaly'].sum()} anomalies.")
    return df

def perform_eda(df):
    """Performs Exploratory Data Analysis and generates visualizations."""
    logging.info("Starting EDA.")
    
    # Data Issues and Fixes (Minimal for synthetic data)
    if df.isnull().any().any():
        logging.warning("Missing values found. Imputing with forward fill.")
        df.fillna(method='ffill', inplace=True)
    else:
        logging.info("No missing values found.")
        
    # Outlier check (will be handled by anomaly detection, but a quick check)
    logging.info("Outlier check: Anomalies are intentionally embedded.")
    
    # Visualization 1: Time Series Plots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)
    features = ['Temperature', 'Pressure', 'Vibration']
    for i, feature in enumerate(features):
        axes[i].plot(df.index, df[feature], label=feature)
        anomalies = df[df['is_anomaly'] == 1]
        axes[i].scatter(anomalies.index, anomalies[feature], color='red', label='Anomaly', s=20)
        axes[i].set_title(f'Time Series of {feature}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_time_series_plots.png'))
    plt.close(fig)
    logging.info("Saved EDA time series plots.")
    
    # Visualization 2: Feature Distributions
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for i, feature in enumerate(features):
        df[feature].plot(kind='hist', bins=30, ax=axes[i], title=f'Distribution of {feature}')
        axes[i].axvline(df[feature].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_feature_distributions.png'))
    plt.close(fig)
    logging.info("Saved EDA feature distribution plots.")
    
    # Visualization 3: Correlation Matrix
    corr_matrix = df[features].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr_matrix, cmap='coolwarm')
    plt.colorbar(cax)
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(features)
    ax.set_yticklabels(features)
    plt.title('Feature Correlation Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_correlation_matrix.png'))
    plt.close(fig)
    logging.info("Saved EDA correlation matrix plot.")
    
    return df

# --- Phase 3: Feature Engineering and Scaling ---

def feature_engineering(df, window=5):
    """
    Creates rolling means, rolling std, and lag features.
    Normalizes/Scales features using MinMaxScaler.
    """
    logging.info("Starting feature engineering.")
    
    features = ['Temperature', 'Pressure', 'Vibration']
    X = df[features].copy()
    
    # 1. Rolling Mean and Std (Temporal features)
    for feature in features:
        X[f'{feature}_roll_mean'] = X[feature].rolling(window=window).mean()
        X[f'{feature}_roll_std'] = X[feature].rolling(window=window).std()
        
    # 2. Lag Features (Sequential features)
    for feature in features:
        X[f'{feature}_lag1'] = X[feature].shift(1)
        
    # 3. Domain-relevant features (Rate of change)
    X['Temp_diff'] = X['Temperature'].diff()
    X['Pressure_diff'] = X['Pressure'].diff()
    
    # Drop NaN values created by rolling/lag features
    X.dropna(inplace=True)
    y = df.loc[X.index, 'is_anomaly']
    
    # 4. Normalize/Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    logging.info(f"Feature engineering complete. New shape: {X_scaled.shape}")
    return X_scaled, y, scaler, X.columns.tolist()

# --- Phase 4: Model 1 (Isolation Forest) ---

def train_and_evaluate_iforest(X_scaled, y):
    """
    Trains an Isolation Forest model and predicts anomalies.
    """
    logging.info("Starting Isolation Forest training and prediction.")
    
    # Isolation Forest is an unsupervised model, so we train on the entire dataset.
    # We assume a contamination rate based on the synthetic data generation (approx 2.1%)
    contamination_rate = y.sum() / len(y)
    
    iforest = IsolationForest(
        contamination=contamination_rate, 
        random_state=42, 
        n_estimators=100, 
        max_samples='auto'
    )
    
    # Fit the model
    iforest.fit(X_scaled)
    
    # Predict anomalies (-1 for anomaly, 1 for normal)
    predictions = iforest.predict(X_scaled)
    
    # Convert predictions to 1 for anomaly, 0 for normal
    anomaly_labels = pd.Series(np.where(predictions == -1, 1, 0), index=X_scaled.index)
    
    # Get anomaly scores (lower score is more anomalous)
    anomaly_scores = pd.Series(iforest.decision_function(X_scaled), index=X_scaled.index)
    
    # Evaluate (since we have labels)
    precision, recall, f1, _ = precision_recall_fscore_support(y, anomaly_labels, average='binary', zero_division=0)
    roc_auc = roc_auc_score(y, -anomaly_scores) # Note: Isolation Forest scores are inverted
    
    results = {
        'model_name': 'Isolation Forest',
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'anomaly_scores': anomaly_scores,
        'anomaly_labels': anomaly_labels,
        'contamination': contamination_rate
    }
    
    logging.info(f"Isolation Forest Results: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")
    return results

# --- Phase 5: Model 2 (Autoencoder) ---

def create_autoencoder(input_dim):
    """Creates a simple Autoencoder model using Keras."""
    input_layer = Input(shape=(input_dim,))
    # Encoder
    encoded = Dense(int(input_dim/2), activation='relu')(input_layer)
    encoded = Dense(int(input_dim/4), activation='relu')(encoded)
    # Decoder
    decoded = Dense(int(input_dim/2), activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_and_evaluate_autoencoder(X_scaled, y):
    """
    Trains the Autoencoder and uses reconstruction error to detect anomalies.
    """
    logging.info("Starting Autoencoder training and prediction.")
    
    input_dim = X_scaled.shape[1]
    autoencoder = create_autoencoder(input_dim)
    
    # Train only on normal data (assuming anomalies are rare)
    X_normal = X_scaled[y == 0]
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=0)
    
    # Train the model
    history = autoencoder.fit(
        X_normal, X_normal,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Get reconstruction error for all data points
    X_pred = autoencoder.predict(X_scaled, verbose=0)
    mse = pd.Series(np.mean(np.power(X_scaled - X_pred, 2), axis=1), index=X_scaled.index)
    
    # Determine anomaly threshold (e.g., 95th percentile of reconstruction error on normal data)
    mse_normal = np.mean(np.power(X_normal - autoencoder.predict(X_normal, verbose=0), 2), axis=1)
    threshold = np.percentile(mse_normal, 95)
    
    # Predict anomalies
    anomaly_labels = (mse > threshold).astype(int)
    
    # Evaluate
    precision, recall, f1, _ = precision_recall_fscore_support(y, anomaly_labels, average='binary', zero_division=0)
    roc_auc = roc_auc_score(y, mse)
    
    results = {
        'model_name': 'Autoencoder',
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'anomaly_scores': mse,
        'anomaly_labels': anomaly_labels,
        'threshold': threshold
    }
    
    logging.info(f"Autoencoder Results: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")
    return results

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # Phase 2 Execution
        df = generate_synthetic_data()
        df = perform_eda(df)
        df.to_csv(os.path.join(OUTPUT_DIR, 'synthetic_data.csv'))
        logging.info("Synthetic data saved and EDA complete.")
        
        # Phase 3 Execution
        X_scaled, y, scaler, feature_names = feature_engineering(df)
        
        # Phase 4 Execution
        iforest_results = train_and_evaluate_iforest(X_scaled, y)
        
        # Phase 5 Execution
        ae_results = train_and_evaluate_autoencoder(X_scaled, y)
        
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        print(f"An error occurred: {e}")

# Placeholder for subsequent phases (will be filled in later)






# --- Phase 6: Model Evaluation and Visualization ---

def plot_anomaly_results(df, iforest_results, ae_results):
    """
    Generates plots showing anomalies on actual time-series graphs.
    """
    logging.info("Starting anomaly results visualization.")
    
    # Re-align indices after feature engineering
    df_results = df.loc[iforest_results['anomaly_labels'].index].copy()
    
    # Add model results to the dataframe
    df_results['iforest_anomaly'] = iforest_results['anomaly_labels']
    df_results['ae_anomaly'] = ae_results['anomaly_labels']
    
    features = ['Temperature', 'Pressure', 'Vibration']
    
    for feature in features:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df_results.index, df_results[feature], label='Value', color='blue', alpha=0.6)
        
        # Plot true anomalies
        true_anomalies = df_results[df_results['is_anomaly'] == 1]
        ax.scatter(true_anomalies.index, true_anomalies[feature], color='black', label='True Anomaly', s=50, marker='x')
        
        # Plot Isolation Forest anomalies
        iforest_anomalies = df_results[df_results['iforest_anomaly'] == 1]
        ax.scatter(iforest_anomalies.index, iforest_anomalies[feature], color='red', label='IF Anomaly', s=20)
        
        # Plot Autoencoder anomalies
        ae_anomalies = df_results[df_results['ae_anomaly'] == 1]
        ax.scatter(ae_anomalies.index, ae_anomalies[feature], color='green', label='AE Anomaly', s=10)
        
        ax.set_title(f'Anomaly Detection Results for {feature}')
        ax.set_xlabel('Time')
        ax.set_ylabel(feature)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'anomaly_results_{feature}.png'))
        plt.close(fig)
        logging.info(f"Saved anomaly results plot for {feature}.")

def generate_summary_data(iforest_results, ae_results):
    """
    Combines model results into a structured format for the summary document.
    """
    summary_data = {
        'Isolation Forest': {
            'Precision': f"{iforest_results['precision']:.4f}",
            'Recall': f"{iforest_results['recall']:.4f}",
            'F1-Score': f"{iforest_results['f1_score']:.4f}",
            'ROC-AUC': f"{iforest_results['roc_auc']:.4f}",
            'Model Type': 'Statistical/Unsupervised',
            'Intuition': 'Uses random trees to isolate anomalies, which are fewer and closer to the root of the tree.',
            'Hyperparameters': f"Contamination={iforest_results['contamination']:.4f}, n_estimators=100"
        },
        'Autoencoder': {
            'Precision': f"{ae_results['precision']:.4f}",
            'Recall': f"{ae_results['recall']:.4f}",
            'F1-Score': f"{ae_results['f1_score']:.4f}",
            'ROC-AUC': f"{ae_results['roc_auc']:.4f}",
            'Model Type': 'Deep Learning',
            'Intuition': 'Learns a compressed representation of normal data; anomalies result in high reconstruction error.',
            'Hyperparameters': f"Threshold={ae_results['threshold']:.4f}, Epochs=50 (early stopped), Architecture=3-layer AE"
        }
    }
    return summary_data

def write_final_documents(df, iforest_results, ae_results, feature_names):
    """
    Generates the Summary Document and README.md.
    """
    logging.info("Generating final documents.")
    
    summary_data = generate_summary_data(iforest_results, ae_results)
    
    # --- README.md ---
    readme_content = f"""# Time Series Anomaly Detection for IoT Sensors

## Project Overview
This project implements an end-to-end pipeline for multivariate time series anomaly detection using synthetic IoT sensor data. It compares a traditional statistical/unsupervised method (Isolation Forest) with a deep learning approach (Autoencoder).

## How to Run the Code

1.  **Clone the repository (Simulated):**
    ```bash
    # git clone <repo_url>
    # cd time_series_anomaly_detection
    ```

2.  **Setup Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install scikit-learn pandas matplotlib tensorflow
    ```

3.  **Execute the Pipeline:**
    ```bash
    python3 src/anomaly_detection_pipeline.py
    ```

## Dependencies
-   Python 3.x
-   `numpy`
-   `pandas`
-   `matplotlib`
-   `scikit-learn`
-   `tensorflow` (for Keras Autoencoder)

## Dataset Structure
The pipeline generates a synthetic dataset with 1000 hourly samples of three sensor readings:
-   `Temperature`
-   `Pressure`
-   `Vibration`
The dataset includes embedded point, contextual, and collective anomalies, labeled in the `is_anomaly` column.

## Output Files
All results are saved in the `output/` directory:
-   `pipeline.log`: Execution log.
-   `synthetic_data.csv`: The generated dataset.
-   `eda_*.png`: Exploratory Data Analysis visualizations.
-   `anomaly_results_*.png`: Plots showing detected anomalies on time series.
-   `summary.md`: Detailed project summary and model comparison.
"""
    
    # --- Summary Document (Markdown) ---
    summary_content = f"""# Time Series Anomaly Detection Project Summary

## 1. Problem Understanding
The objective was to build a complete anomaly detection pipeline for multivariate IoT sensor time series data. The pipeline covers data preparation, feature engineering, model implementation, and comparative evaluation. Anomalies in IoT data are critical as they often indicate sensor malfunction, system failure, or security breaches.

## 2. Approach and Data
A synthetic dataset of 1000 hourly samples for Temperature, Pressure, and Vibration was generated, including three types of labeled anomalies (point, contextual, collective). This approach ensures a ground truth for rigorous model evaluation.

### Data Issues and Fixes
-   **Missing Values:** None in the synthetic data, but a forward-fill imputation strategy was included in the `perform_eda` function for robustness.
-   **Outliers:** Anomalies were intentionally embedded and handled by the detection models rather than pre-processed removal.

## 3. Feature Engineering
The raw sensor data was augmented with temporal and domain-relevant features to capture time-series characteristics:
| Feature | Justification |
| :--- | :--- |
| **Rolling Mean (Window=5)** | Captures short-term trend and smooths noise. |
| **Rolling Standard Deviation (Window=5)** | Captures short-term volatility and variance. |
| **Lag-1 Features** | Captures immediate temporal dependency (autocorrelation). |
| **Rate of Change (Diff)** | Captures sudden shifts in sensor behavior, indicating potential point anomalies. |

All features were scaled using `MinMaxScaler` to ensure equal contribution to the distance-based and neural network models.

## 4. Model Choices and Intuition

| Model | Type | Intuition | Architecture/Logic |
| :--- | :--- | :--- | :--- |
| **Isolation Forest** | Statistical/Unsupervised | Anomalies are "isolated" faster in a random tree structure because they are few and different. The model measures the number of splits required to isolate a point. | Ensemble of decision trees. Anomaly score is based on path length. |
| **Autoencoder** | Deep Learning | Learns the compressed, low-dimensional representation of *normal* data. Anomalies, being unseen during training, are poorly reconstructed, resulting in a high reconstruction error (MSE). | 3-layer fully connected network (Encoder-Decoder) trained on normal data only. |

## 5. Model Comparison and Evaluation

The models were evaluated using the ground truth labels.

| Metric | Isolation Forest | Autoencoder |
| :--- | :--- | :--- |
| **Precision** | {summary_data['Isolation Forest']['Precision']} | {summary_data['Autoencoder']['Precision']} |
| **Recall** | {summary_data['Isolation Forest']['Recall']} | {summary_data['Autoencoder']['Recall']} |
| **F1-Score** | {summary_data['Isolation Forest']['F1-Score']} | {summary_data['Autoencoder']['F1-Score']} |
| **ROC-AUC** | {summary_data['Isolation Forest']['ROC-AUC']} | {summary_data['Autoencoder']['ROC-AUC']} |

### Key Insights and Limitations
-   **Isolation Forest** generally offers a good balance of speed and performance for high-dimensional data. Its performance is highly dependent on the initial `contamination` estimate.
-   **Autoencoder** is excellent at detecting anomalies that deviate significantly from the learned normal patterns (reconstruction error). Its training is more computationally intensive and requires careful selection of the reconstruction error threshold.
-   **Limitation:** The synthetic data, while useful for demonstration, may not fully capture the complexity of real-world sensor noise and drift. Future work should involve testing on a real-world dataset (e.g., NASA Bearing) and exploring sequence-aware models like LSTM Autoencoders.

## 6. Visualizations
See the `output/` directory for the following plots:
-   `eda_time_series_plots.png`
-   `eda_feature_distributions.png`
-   `eda_correlation_matrix.png`
-   `anomaly_results_Temperature.png`
-   `anomaly_results_Pressure.png`
-   `anomaly_results_Vibration.png`
"""
    
    with open(os.path.join(OUTPUT_DIR, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    with open(os.path.join(OUTPUT_DIR, 'summary.md'), 'w') as f:
        f.write(summary_content)
    
    logging.info("Final documents (README.md and summary.md) generated.")

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # Phase 2 Execution
        df = generate_synthetic_data()
        df = perform_eda(df)
        df.to_csv(os.path.join(OUTPUT_DIR, 'synthetic_data.csv'))
        logging.info("Synthetic data saved and EDA complete.")
        
        # Phase 3 Execution
        X_scaled, y, scaler, feature_names = feature_engineering(df)
        
        # Phase 4 Execution
        iforest_results = train_and_evaluate_iforest(X_scaled, y)
        
        # Phase 5 Execution
        ae_results = train_and_evaluate_autoencoder(X_scaled, y)
        
        # Phase 6 Execution
        plot_anomaly_results(df, iforest_results, ae_results)
        summary_data = generate_summary_data(iforest_results, ae_results)
        write_final_documents(df, iforest_results, ae_results, feature_names)
        
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        print(f"An error occurred: {e}")
