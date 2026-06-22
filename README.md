# NYC Taxi Big Data Analysis

A comprehensive big data analytics project that leverages **Apache Spark** and **TensorFlow** to analyze nearly **10 million** NYC Yellow and Green taxi trip records from Q1 2024. The project covers the full data science pipeline, from ingestion and preprocessing to advanced machine learning and deep learning.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Features and Pipeline](#features-and-pipeline)
- [Key Results](#key-results)
- [Getting Started](#getting-started)
- [Team](#team)
- [License](#license)

---

## Overview

This project demonstrates an end-to-end big data workflow using real-world NYC taxi trip data. We process **9.7 million+ combined records** across Yellow and Green taxi services, engineer meaningful features, and apply a spectrum of machine learning techniques, from classical regression and classification to unsupervised clustering and neural networks.

The analysis provides actionable insights into urban transportation patterns, fare dynamics, tipping behavior, and trip segmentation across New York City.

---

## Dataset

| Source | Records | Period |
|--------|---------|--------|
| Yellow Taxi | ~9.55M | Jan-Mar 2024 |
| Green Taxi | ~168K | Jan-Mar 2024 |
| Taxi Zone Lookup | 265 zones | - |

**Data Source:** [NYC Taxi & Limousine Commission (TLC) Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

All data files are in **Parquet** format (taxi trips) and **CSV** (zone lookup), totaling approximately **9.72 million records** after merging.

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Big Data Processing** | Apache Spark (PySpark 4.1.x) |
| **Machine Learning** | Spark MLlib (Regression, Classification, Clustering) |
| **Deep Learning** | TensorFlow / Keras |
| **Data Manipulation** | Pandas, NumPy, PyArrow |
| **Visualization** | Matplotlib, Seaborn |
| **Evaluation** | Scikit-learn |
| **Runtime** | Java 17 (OpenJDK Temurin) |

---

## Project Structure

```
NYC-Taxi-Bigdata-Analysis/
├── nyc_taxi_project.ipynb   # Main analysis notebook (full pipeline)
└── README.md                # Project documentation
```

> **Note:** The raw data files (`.parquet` and `.csv`) are not included in this repository due to their size. See [Getting Started](#getting-started) for download instructions.

---

## Features and Pipeline

### 0. Environment Setup
- Automated dependency installation (`pyspark`, `pandas`, `pyarrow`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`)
- Java 17 runtime verification
- Spark session configuration (4 GB driver memory, 50 shuffle partitions)

### 1. Data Loading
- Loads 3 months of Yellow Taxi and 3 months of Green Taxi Parquet files
- Loads the Taxi Zone Lookup CSV for geographic enrichment
- Harmonizes differing schemas between Yellow (`tpep_*`) and Green (`lpep_*`) taxi data
- Unions both datasets into a single unified DataFrame

### 2. Data Preprocessing
- **Null analysis** across all columns, identifying ~760K nulls in `passenger_count` and `RatecodeID`
- **Outlier filtering:**
  - Trip distance: 0-100 miles
  - Fare amount: $0-$500
  - Passenger count: 1-6
  - Valid payment types only (1-4)
- After cleaning: **~8.63M records** retained

### 3. Feature Engineering
- `trip_duration_min` - calculated from pickup/dropoff timestamps
- `pickup_hour`, `pickup_dayofweek`, `pickup_month` - temporal decomposition
- `is_weekend` - binary weekend flag
- `speed_mph` - derived from distance and duration
- `tip_percentage` - tip as a percentage of fare
- `high_tip` - binary label (tip > 15%)
- `taxi_type_idx` - encoded taxi type
- Additional filters remove unrealistic durations (< 1 min or > 180 min) and speeds (> 80 mph)
- Final dataset: **~8.59M records**

### 4. Data Visualization
A 0.5% stratified sample (~43K rows) is used for plotting:

- Distribution histograms (distance, fare, tip, duration, speed, passengers)
- Hourly demand patterns (trip count by hour of day)
- Correlation heatmap across all numeric features
- Box plots by payment type
- Yellow vs. Green taxi comparative analysis

### 5. Supervised Learning - Regression
**Objective:** Predict `fare_amount` from trip characteristics

| Model | RMSE | R² |
|-------|------|-----|
| Linear Regression | 5.16 | 0.908 |
| Random Forest Regressor | 4.82 | 0.920 |

Features: `trip_distance`, `trip_duration_min`, `pickup_hour`, `pickup_dayofweek`, `passenger_count`, `is_weekend`, `taxi_type_idx`

### 6. Supervised Learning - Classification
**Objective:** Predict `high_tip` (> 15%) for credit card payments

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Random Forest Classifier | 0.7651 | 0.7648 |

Includes feature importance visualization identifying `fare_amount`, `trip_distance`, and `trip_duration_min` as top predictors.

### 7. Unsupervised Learning - K-Means Clustering
- Elbow method analysis for K = 2 through 8
- Optimal **K = 4** selected based on WSSSE curve
- Clustering features: `trip_distance`, `trip_duration_min`, `fare_amount`, `pickup_hour`, `tip_percentage`
- Cluster profiles visualized across all features with labeled interpretations

### 8. Deep Learning - Neural Network
**Objective:** Fare prediction using a TensorFlow/Keras sequential model

**Architecture:**
```
Dense(128, ReLU) -> BatchNorm -> Dropout(0.3)
Dense(64, ReLU)  -> BatchNorm -> Dropout(0.2)
Dense(32, ReLU)  -> Dense(1, Linear)
```

| Metric | Value |
|--------|-------|
| Total Parameters | 12,161 |
| Test RMSE | 4.64 |
| Test R² | 0.925 |

Trained with Adam optimizer, MSE loss, early stopping, and learning rate reduction on plateau.

---

## Key Results

| Task | Best Model | Key Metric |
|------|-----------|------------|
| Fare Prediction (ML) | Random Forest Regressor | R² = 0.920 |
| Fare Prediction (DL) | Neural Network | R² = 0.925 |
| High Tip Classification | Random Forest Classifier | Accuracy = 76.5% |
| Trip Segmentation | K-Means (K=4) | 4 distinct trip profiles |

**Key Insights:**
- Trip distance and duration are the strongest fare predictors
- The neural network slightly outperforms traditional ML for fare regression
- Tipping behavior is most influenced by fare amount and trip distance
- K-Means reveals 4 natural trip segments: short urban, medium urban, long/airport, and an outlier cluster

---

## Getting Started

### Prerequisites
- **Python 3.9+**
- **Java 17** (OpenJDK recommended)
- **Jupyter Notebook** or **JupyterLab**

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Skapos-crypto/NYC-Taxi-Bigdata-Analysis.git
   cd NYC-Taxi-Bigdata-Analysis
   ```

2. **Download the data** from the [NYC TLC Trip Record Data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page):
   - `yellow_tripdata_2024-01.parquet`
   - `yellow_tripdata_2024-02.parquet`
   - `yellow_tripdata_2024-03.parquet`
   - `green_tripdata_2024-01.parquet`
   - `green_tripdata_2024-02.parquet`
   - `green_tripdata_2024-03.parquet`
   - `taxi_zone_lookup.csv`

3. **Update the data path** in the notebook:
   ```python
   DATA_PATH = r'C:\path\to\your\data'
   ```

4. **Launch and run the notebook:**
   ```bash
   jupyter notebook nyc_taxi_project.ipynb
   ```
   > Dependencies are automatically installed in the first cell.

---

## Team

This project was developed as a collaborative Big Data course project.

---

## License

This project is open source and available under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

All derivative works must also be distributed under the same license terms.
