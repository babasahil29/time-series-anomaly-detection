# Time Series Anomaly Detection Project Summary

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
| **Precision** | 0.7521 | 0.7047 |
| **Recall** | 0.7521 | 0.8678 |
| **F1-Score** | 0.7521 | 0.7778 |
| **ROC-AUC** | 0.9560 | 0.9782 |

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
