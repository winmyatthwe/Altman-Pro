# Bankruptcy Prediction — Big Data Assignment 2

A machine learning project that predicts corporate bankruptcy using financial ratio data from 6,819 Taiwanese companies. Five models are trained, evaluated, and deployed in an interactive Shiny dashboard.

## How to Run

### 1. Install R dependencies

```r
install.packages(c("shiny", "shinydashboard", "dplyr", "ggplot2",
                   "pROC", "randomForest", "DT", "tidyr", "caret",
                   "kernlab", "ada", "DMwR2", "keras3", "tensorflow"))
```

### 2. Run scripts in order

```r
source("section1_data_quality.R")   # generates summary stats and plots
source("data_prep.R")               # creates df_final.rds, train_scaled.rds,
                                    # test_scaled.rds, scaler.rds
source("model1_logistic_regression.R")  # creates lr_results.rds
source("model2_random_forest.R")        # creates rf_results.rds
source("model3_neural_network.R")       # creates nn_results.rds
source("model4_svm.R")                  # creates svm_results.rds
source("model5_adaboost.R")             # creates ada_results.rds
source("model6_deep_nn_keras.R")        # creates dnn_keras_model.keras
                                        # and dnn_results.rds
```

### 3. Launch the dashboard

```r
shiny::runApp("final_dashboard.R")
```

> **Note:** The Keras DNN model is saved as `dnn_keras_model.keras` (not `.rds`) because Keras models cannot be serialised with `saveRDS`. Make sure this file is in the same folder as `final_dashboard.R`.

---

## Dashboard Features

The Shiny dashboard has 7 tabs:

1. **Background** — model descriptions and dataset overview
2. **Steps Performed** — full methodology walkthrough
3. **How We Did It** — SMOTE, scaling, and tuning decisions
4. **Pros & Cons** — trade-offs of each model
5. **Results** — confusion matrix, ROC curve, and metrics per model
6. **Model Comparison** — combined ROC curves and head-to-head metrics
7. **Live Prediction** — enter financial ratios and get real-time predictions from all 5 models, with stakeholder-specific advice (Lender / Investor / Regulator)

