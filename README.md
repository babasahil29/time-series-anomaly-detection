# Time Series Anomaly Detection for IoT Sensors

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
