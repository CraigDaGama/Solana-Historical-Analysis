# 📈 Stock Trend Prediction with Walk-Forward Validation

A machine learning project that predicts **Solana (SOL)** cryptocurrency movement — classified as `Up`, `Down`, or `Flat` — using walk-forward validation across multiple years. This project evaluates **Random Forest** and **XGBoost** classifiers with engineered features, targeting improved accuracy over time.

---

## 📂 Dataset

**Source**: [Solana Historical Data on Kaggle](https://www.kaggle.com/datasets/craigdagama/solana-historical-data/data)

* This dataset contains daily OHLCV data for Solana from 2020 through 2025.
* I created this dataset myself since earlier Solana datasets only covered single years, restricting deeper market analysis. This dataset provides a broader timeframe, enabling more insightful long-term trend evaluation.

---

## 🔧 Feature Engineering

After cleaning and transforming the dataset, the following features were created:

* 📉 `MA7`, `MA21`: 7-day and 21-day moving averages
* 🔁 `Daily_Return`: Day-over-day percentage price change
* ⚡ `Volatility`: 7-day rolling standard deviation of returns
* 🔙 `Close_Lag1`, `Volume_Lag1`: Previous day's close and volume
* 🎯 `Target_dynamic`: Categorical label with 3 classes:

  * **Up**, **Down**, **Flat** — based on return thresholds

---

## 🔁 Walk-Forward Validation Strategy

To simulate a real-world deployment, we train on past years and test on the next:

| Train Years | Test Year |
| ----------- | --------- |
| 2020–2022   | 2023      |
| 2020–2023   | 2024      |
| 2020–2024   | 2025      |

---

## 📊 Accuracy Progression

### 🟠 Initial Model – Random Forest

| Year | Accuracy | Notes                                          |
| ---- | -------- | ---------------------------------------------- |
| 2023 | 67.67%   | Strong Flat class; Down class poorly detected  |
| 2024 | 67.21%   | Minimal gains; Up class still weak             |
| 2025 | 77.64%   | Noticeable improvement with more training data |

### 🔄 Upgrade – XGBoost Classifier

To overcome the limitations of Random Forest, especially with class imbalance and temporal complexity, **XGBoost** was used.

| Year | Accuracy   | Highlights                                      |
| ---- | ---------- | ----------------------------------------------- |
| 2023 | 95.62%     | All classes well detected; massive jump from RF |
| 2024 | 96.72%     | Near-perfect on Flat class; improved Up/Down    |
| 2025 | **99.38%** | Perfect or near-perfect classification          |

✅ **Key Insight**: XGBoost boosted final year accuracy by **\~22%** compared to Random Forest (77.64% → 99.38%).

---

## 🆚 Model Comparison (2025)

| Metric   | Random Forest | XGBoost |
| -------- | ------------- | ------- |
| Accuracy | 0.7764        | 0.9938  |
| Down F1  | 0.36          | 0.98    |
| Flat F1  | 0.86          | 1.00    |
| Up F1    | 0.55          | 1.00    |

---

## 📈 Evaluation & Visuals

* ✅ **Classification Reports**: For all test years
* ✅ **Confusion Matrices**: Plotted using `ConfusionMatrixDisplay`
* ✅ **Walk-Forward Accuracy Chart**:

  ```
  2023: 0.9562
  2024: 0.9672
  2025: 0.9938
  ```

---

## 💡 Key Learnings

* **Feature Engineering** was crucial — especially lagged and rolling features.
* **XGBoost outperformed Random Forest** significantly, especially with imbalanced classes.
* **Temporal validation** helps uncover overfitting that’s hidden in random splits.

---

## 🛠 Tools & Libraries

* Python (pandas, numpy)
* scikit-learn
* XGBoost
* seaborn, matplotlib

---

