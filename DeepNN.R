# =============================================================================
# MODEL 4: DEEP NEURAL NETWORK (KERAS / TENSORFLOW)
# Big Data Management Module — 07 36320
# Bankruptcy Prediction — Taiwan Economic Journal (TEJ) Dataset
# =============================================================================
# Deep Learning is a subfield of machine learning using neural networks with
# multiple layers of non-linear transformations to learn hierarchical feature
# representations from data (LeCun et al., 2015; Goodfellow et al., 2016).
# This model satisfies LO1 (Deep Learning Techniques) through:
#   - Three hidden layers with ReLU activations
#   - Batch Normalisation for training stability (Ioffe & Szegedy, 2015)
#   - Dropout regularisation as implicit ensembling (Srivastava et al., 2014)
#   - Adam optimiser with binary cross-entropy loss
#   - Early stopping to prevent overfitting
#
# ARCHITECTURE:
#   Input (22 features)
#   -> Dense(64, ReLU) + BatchNorm + Dropout(0.4)
#   -> Dense(32, ReLU) + BatchNorm + Dropout(0.3)
#   -> Dense(16, ReLU)             + Dropout(0.2)
#   -> Dense(1, Sigmoid)    <- binary output: P(Bankrupt = 1)
#
# Run AFTER: feture_selection.R (which creates train_scaled.rds, test_scaled.rds)
# =============================================================================

# =============================================================================
# STEP 0: INSTALL AND LOAD PACKAGES
# =============================================================================

packages <- c("dplyr", "caret", "themis", "pROC", "ggplot2",
              "keras3", "tensorflow", "tidyr")

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

library(dplyr)
library(caret)
library(themis)    # SMOTE within caret resampling
library(pROC)
library(ggplot2)
library(keras3)    # Keras 3 R interface (TensorFlow backend)
library(tensorflow)
library(tidyr)

# Install TensorFlow backend if not already present
# (Run this once — it downloads ~500MB)
# keras3::install_keras(backend = "tensorflow")

# Fix random seed for reproducibility
set.seed(42)
tensorflow::set_random_seed(42)

# =============================================================================
# STEP 1: LOAD PREPARED DATA
# =============================================================================

cat("=============================================\n")
cat("MODEL 4: DEEP NEURAL NETWORK\n")
cat("Big Data Management — 07 36320\n")
cat("=============================================\n\n")

train <- readRDS("train_scaled.rds")
test  <- readRDS("test_scaled.rds")

# Ensure factor levels are set correctly
levels(train$Bankrupt) <- c("No", "Yes")
levels(test$Bankrupt)  <- c("No", "Yes")

cat("Training set:", nrow(train), "rows |",
    "Class distribution:", paste(table(train$Bankrupt), collapse = " / "), "\n")
cat("Test set    :", nrow(test),  "rows |",
    "Class distribution:", paste(table(test$Bankrupt),  collapse = " / "), "\n\n")

n_features <- ncol(train) - 1
cat("Number of features:", n_features, "\n\n")

# =============================================================================
# STEP 2: CROSS-VALIDATION SETUP WITH SMOTE
# =============================================================================
# We use 10-fold stratified cross-validation with SMOTE applied only to
# training folds — identical to the LR and RF models — ensuring a fair
# comparison. SMOTE is handled by caret's themis integration via
# sampling = "smote" in trainControl.

cat("=============================================\n")
cat("STEP 2: CROSS-VALIDATION SETUP\n")
cat("=============================================\n\n")

cat("Configuration:\n")
cat("  Method     : 10-fold stratified cross-validation\n")
cat("  Resampling : SMOTE (training folds only)\n")
cat("  Metric     : ROC (AUC-ROC)\n\n")

cv_control <- trainControl(
  method          = "cv",
  number          = 10,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  sampling        = "smote"   # SMOTE applied within each training fold
)

# =============================================================================
# STEP 3: DEFINE DNN ARCHITECTURE (Keras model builder)
# =============================================================================
# keras3 integrates with caret via a custom model specification.
# The build_dnn() function is called once per CV fold and once for the
# final model — ensuring consistent architecture across all runs.

cat("=============================================\n")
cat("STEP 3: DNN ARCHITECTURE\n")
cat("=============================================\n\n")

cat("Architecture:\n")
cat("  Input       : 22 features\n")
cat("  Hidden 1    : Dense(64) + BatchNorm + ReLU + Dropout(0.4)\n")
cat("  Hidden 2    : Dense(32) + BatchNorm + ReLU + Dropout(0.3)\n")
cat("  Hidden 3    : Dense(16)             + ReLU + Dropout(0.2)\n")
cat("  Output      : Dense(1, Sigmoid)\n")
cat("  Optimiser   : Adam (lr = 0.001)\n")
cat("  Loss        : Binary Cross-Entropy\n")
cat("  Regulariser : L2 (lambda = 1e-4) on all Dense layers\n")
cat("  Early stop  : patience = 10 (monitor val_loss)\n\n")

build_dnn <- function(n_feat) {
  model <- keras_model_sequential(name = "DNN_Bankruptcy") %>%
    
    # --- Hidden Layer 1 -------------------------------------------------------
  layer_dense(
    units              = 64,
    kernel_regularizer = regularizer_l2(1e-4),
    input_shape        = n_feat
  ) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_dropout(rate = 0.4) %>%
    
    # --- Hidden Layer 2 -------------------------------------------------------
  layer_dense(
    units              = 32,
    kernel_regularizer = regularizer_l2(1e-4)
  ) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_dropout(rate = 0.3) %>%
    
    # --- Hidden Layer 3 -------------------------------------------------------
  layer_dense(
    units              = 16,
    kernel_regularizer = regularizer_l2(1e-4)
  ) %>%
    layer_activation("relu") %>%
    layer_dropout(rate = 0.2) %>%
    
    # --- Output Layer ---------------------------------------------------------
  layer_dense(units = 1, activation = "sigmoid")
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss      = "binary_crossentropy",
    metrics   = list(metric_auc(name = "auc"))
  )
  
  return(model)
}

# Print model summary
dnn_summary_model <- build_dnn(n_features)
summary(dnn_summary_model)
cat("\n")

# =============================================================================
# STEP 4: TRAIN MODEL USING 10-FOLD CV
# =============================================================================
# We use caret's train() wrapper with a custom Keras method definition.
# This keeps the evaluation framework identical to LR and RF models.
# Early stopping (patience = 10) prevents overfitting on the minority class.

cat("=============================================\n")
cat("STEP 4: TRAINING WITH 10-FOLD CV\n")
cat("=============================================\n")
cat("This may take 5-10 minutes depending on hardware.\n\n")

# --- Define custom caret model method for Keras DNN --------------------------
keras_dnn_method <- list(
  label = "Deep Neural Network (Keras)",
  library = c("keras3", "tensorflow"),
  type = "Classification",
  
  parameters = data.frame(
    parameter = "epochs",
    class     = "numeric",
    label     = "Max Epochs"
  ),
  
  grid = function(x, y, len = NULL, search = "grid") {
    data.frame(epochs = 80)
  },
  
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    # Convert factor to binary 0/1
    y_bin <- as.integer(y == lev[2])
    
    # Class weights for imbalance (even after SMOTE, slight residual imbalance)
    n_neg   <- sum(y_bin == 0)
    n_pos   <- sum(y_bin == 1)
    cw      <- if (n_pos > 0) n_neg / n_pos else 1.0
    class_w <- list("0" = 1.0, "1" = cw)
    
    # Build a fresh model for this fold
    model <- build_dnn(ncol(x))
    
    # Early stopping callback
    early_stop <- callback_early_stopping(
      monitor              = "val_loss",
      patience             = 10,
      restore_best_weights = TRUE,
      verbose              = 0
    )
    
    # Train
    model %>% fit(
      x                = as.matrix(x),
      y                = y_bin,
      epochs           = param$epochs,
      batch_size       = 256,
      validation_split = 0.1,
      class_weight     = class_w,
      callbacks        = list(early_stop),
      verbose          = 0
    )
    
    return(model)
  },
  
  predict = function(modelFit, newdata, submodels = NULL) {
    probs <- as.numeric(predict(modelFit, as.matrix(newdata)))
    ifelse(probs >= 0.5, "Yes", "No")
  },
  
  prob = function(modelFit, newdata, submodels = NULL) {
    p_yes <- as.numeric(predict(modelFit, as.matrix(newdata)))
    data.frame(No = 1 - p_yes, Yes = p_yes)
  },
  
  sort = function(x) x[order(x$epochs), ]
)

# --- Feature matrix for caret ------------------------------------------------
# caret's train() needs X (matrix) and Y (factor) separately when using
# a custom method. Extract from the training dataframe.
X_train <- train %>% select(-Bankrupt) %>% as.matrix()
Y_train <- train$Bankrupt

# --- Run cross-validation training -------------------------------------------
set.seed(42)

dnn_cv <- train(
  x         = X_train,
  y         = Y_train,
  method    = keras_dnn_method,
  trControl = cv_control,
  metric    = "ROC",
  tuneGrid  = data.frame(epochs = 80)
)

cat("\nCross-validation complete.\n")
cat("CV AUC-ROC:", round(dnn_cv$results$ROC, 4), "\n")
cat("CV Sensitivity (Recall):", round(dnn_cv$results$Sens, 4), "\n")
cat("CV Specificity:", round(dnn_cv$results$Spec, 4), "\n\n")

# =============================================================================
# STEP 5: TRAIN FINAL MODEL ON FULL TRAINING SET
# =============================================================================
# After CV selection, retrain on the full (SMOTE-augmented) training set
# to produce the final model for test-set evaluation.

cat("=============================================\n")
cat("STEP 5: TRAINING FINAL MODEL\n")
cat("=============================================\n")
cat("Training final DNN on full training set...\n\n")

set.seed(42)

# SMOTE the full training set once for the final model
smote_recipe <- recipe(Bankrupt ~ ., data = train) %>%
  step_smote(Bankrupt, over_ratio = 1, seed = 42)

train_smote <- smote_recipe %>%
  prep(training = train) %>%
  bake(new_data = NULL)

cat("Training set after SMOTE:", nrow(train_smote), "rows |",
    paste(table(train_smote$Bankrupt), collapse = " / "), "\n\n")

X_train_sm <- train_smote %>% select(-Bankrupt) %>% as.matrix()
Y_train_sm <- as.integer(train_smote$Bankrupt == "Yes")

# Compute class weight
n_neg_sm <- sum(Y_train_sm == 0)
n_pos_sm <- sum(Y_train_sm == 1)
cw_final <- list("0" = 1.0, "1" = n_neg_sm / n_pos_sm)

# Build and train final model
dnn_final <- build_dnn(n_features)

early_stop_final <- callback_early_stopping(
  monitor              = "val_loss",
  patience             = 10,
  restore_best_weights = TRUE,
  verbose              = 1
)

history <- dnn_final %>% fit(
  x                = X_train_sm,
  y                = Y_train_sm,
  epochs           = 100,
  batch_size       = 256,
  validation_split = 0.1,
  class_weight     = cw_final,
  callbacks        = list(early_stop_final),
  verbose          = 1
)

cat("\nFinal model training complete.\n")
cat("Epochs run (with early stopping):", length(history$metrics$loss), "\n\n")

# =============================================================================
# STEP 6: PREDICTIONS ON TEST SET
# =============================================================================

cat("=============================================\n")
cat("STEP 6: PREDICTIONS ON TEST SET\n")
cat("=============================================\n\n")

X_test <- test %>% select(-Bankrupt) %>% as.matrix()

# Probability predictions
dnn_pred_prob_raw <- predict(dnn_final, X_test)
dnn_pred_prob     <- as.numeric(dnn_pred_prob_raw)

# Class predictions at 0.5 threshold
dnn_pred_class <- factor(
  ifelse(dnn_pred_prob >= 0.5, "Yes", "No"),
  levels = c("No", "Yes")
)

cat("Prediction distribution:\n")
print(table(dnn_pred_class))
cat("\n")

# =============================================================================
# STEP 7: MODEL EVALUATION
# =============================================================================

cat("=============================================\n")
cat("STEP 7: MODEL EVALUATION\n")
cat("=============================================\n\n")

# --- Confusion Matrix --------------------------------------------------------
cm <- confusionMatrix(
  data      = dnn_pred_class,
  reference = test$Bankrupt,
  positive  = "Yes"
)

cat("Confusion Matrix:\n")
print(cm$table)
cat("\n")

# Extract metrics
accuracy    <- round(cm$overall["Accuracy"]    * 100, 2)
precision   <- round(cm$byClass["Precision"]   * 100, 2)
recall      <- round(cm$byClass["Recall"]      * 100, 2)
f1          <- round(cm$byClass["F1"]          * 100, 2)
specificity <- round(cm$byClass["Specificity"] * 100, 2)

cat(sprintf("Accuracy    : %.2f%%\n", accuracy))
cat(sprintf("Precision   : %.2f%%\n", precision))
cat(sprintf("Recall      : %.2f%%\n", recall))
cat(sprintf("F1 Score    : %.2f%%\n", f1))
cat(sprintf("Specificity : %.2f%%\n", specificity))

# --- AUC-ROC -----------------------------------------------------------------
roc_obj <- roc(
  response  = test$Bankrupt,
  predictor = dnn_pred_prob,
  levels    = c("No", "Yes")
)
auc_val <- round(auc(roc_obj), 4)
cat(sprintf("AUC-ROC     : %.4f\n", auc_val))

# =============================================================================
# STEP 8: VISUALISATIONS
# =============================================================================

cat("\n=============================================\n")
cat("STEP 8: VISUALISATIONS\n")
cat("=============================================\n\n")

# --- Plot 1: Confusion Matrix Heatmap ----------------------------------------
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Predicted", "Actual", "Count")

p_cm <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = Count), size = 8, fontface = "bold", colour = "white") +
  scale_fill_gradient(low = "#8E44AD", high = "#4A235A") +
  labs(
    title    = "Deep Neural Network — Confusion Matrix",
    subtitle = "Test set predictions (threshold = 0.50)",
    x = "Predicted Class", y = "Actual Class"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("plot_16_dnn_confusion_matrix.png",
       plot = p_cm, width = 6, height = 5, dpi = 150)
cat(">> Saved: plot_16_dnn_confusion_matrix.png\n")

# --- Plot 2: ROC Curve -------------------------------------------------------
roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(colour = "#8E44AD", linewidth = 1.4) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  annotate(
    "text", x = 0.65, y = 0.15,
    label    = paste0("AUC = ", auc_val),
    size     = 5, colour = "#8E44AD", fontface = "bold"
  ) +
  labs(
    title    = "Deep Neural Network — ROC Curve",
    subtitle = "Test set performance (10-fold CV final model)",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal(base_size = 13)

ggsave("plot_17_dnn_roc_curve.png",
       plot = p_roc, width = 6, height = 5, dpi = 150)
cat(">> Saved: plot_17_dnn_roc_curve.png\n")

# --- Plot 3: Training History (Loss) ----------------------------------------
history_df <- data.frame(
  Epoch    = seq_along(history$metrics$loss),
  Train    = history$metrics$loss,
  Validate = history$metrics$val_loss
) %>%
  pivot_longer(cols = c("Train", "Validate"),
               names_to = "Set", values_to = "Loss")

p_loss <- ggplot(history_df, aes(x = Epoch, y = Loss, colour = Set)) +
  geom_line(linewidth = 1.2) +
  scale_colour_manual(values = c("Train" = "#2E75B6", "Validate" = "#E74C3C")) +
  labs(
    title    = "Deep Neural Network — Training History",
    subtitle = "Binary cross-entropy loss (early stopping shown by convergence)",
    x = "Epoch", y = "Loss (Binary Cross-Entropy)", colour = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

ggsave("plot_18_dnn_training_loss.png",
       plot = p_loss, width = 8, height = 5, dpi = 150)
cat(">> Saved: plot_18_dnn_training_loss.png\n")

# --- Plot 4: Training History (AUC) -----------------------------------------
auc_df <- data.frame(
  Epoch    = seq_along(history$metrics$auc),
  Train    = history$metrics$auc,
  Validate = history$metrics$val_auc
) %>%
  pivot_longer(cols = c("Train", "Validate"),
               names_to = "Set", values_to = "AUC")

p_auc <- ggplot(auc_df, aes(x = Epoch, y = AUC, colour = Set)) +
  geom_line(linewidth = 1.2) +
  scale_colour_manual(values = c("Train" = "#2E75B6", "Validate" = "#E74C3C")) +
  labs(
    title    = "Deep Neural Network — AUC-ROC During Training",
    subtitle = "Validation AUC plateau signals early stopping",
    x = "Epoch", y = "AUC-ROC", colour = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

ggsave("plot_19_dnn_training_auc.png",
       plot = p_auc, width = 8, height = 5, dpi = 150)
cat(">> Saved: plot_19_dnn_training_auc.png\n")

# --- Plot 5: Four-Model Comparison Bar Chart ---------------------------------
# Load existing model metrics for comparison
lr_results <- readRDS("lr_results.rds")
rf_results <- readRDS("rf_results.rds")
nn_results <- readRDS("nn_results.rds")

comparison_df <- bind_rows(
  lr_results$metrics,
  rf_results$metrics,
  nn_results$metrics,
  data.frame(
    Model       = "Deep Neural Network",
    Accuracy    = accuracy,
    Precision   = precision,
    Recall      = recall,
    F1_Score    = f1,
    Specificity = specificity,
    AUC_ROC     = auc_val * 100   # multiply for chart scale consistency
  )
)

# Pivot for grouped bar chart
comp_long <- comparison_df %>%
  select(Model, Accuracy, Recall, AUC_ROC) %>%
  pivot_longer(cols = c("Accuracy", "Recall", "AUC_ROC"),
               names_to = "Metric", values_to = "Value") %>%
  mutate(
    Metric = recode(Metric,
                    "Accuracy" = "Accuracy",
                    "Recall"   = "Recall",
                    "AUC_ROC"  = "AUC-ROC x100"),
    Model = factor(Model, levels = c(
      "Logistic Regression",
      "Random Forest",
      "Neural Network",
      "Deep Neural Network"
    ))
  )

p_comp <- ggplot(comp_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.75),
           width = 0.65, alpha = 0.88) +
  geom_text(
    aes(label = sprintf("%.1f", Value)),
    position = position_dodge(width = 0.75),
    vjust = -0.4, size = 3.2, fontface = "bold"
  ) +
  scale_fill_manual(values = c(
    "Accuracy"     = "#2980B9",
    "Recall"       = "#27AE60",
    "AUC-ROC x100" = "#8E44AD"
  )) +
  scale_x_discrete(labels = c(
    "Logistic Regression" = "Logistic\nRegression",
    "Random Forest"       = "Random\nForest",
    "Neural Network"      = "Neural Network\n(1-layer)",
    "Deep Neural Network" = "Deep Neural\nNetwork ★ DL"
  )) +
  ylim(0, 108) +
  labs(
    title    = "Four-Model Comparison: Accuracy, Recall, AUC-ROC",
    subtitle = "10-fold stratified CV | TEJ Bankruptcy Dataset | ★ = Deep Learning",
    x = NULL, y = "Score (%)", fill = "Metric"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        axis.text.x     = element_text(size = 10))

ggsave("plot_20_four_model_comparison.png",
       plot = p_comp, width = 10, height = 5.5, dpi = 150)
cat(">> Saved: plot_20_four_model_comparison.png\n")

# =============================================================================
# STEP 9: SAVE RESULTS FOR MODEL COMPARISON
# =============================================================================
# Saves in the same RDS format as lr_results.rds and rf_results.rds so that
# model_comparison.R can load it without modification.

cat("\n=============================================\n")
cat("STEP 9: SAVE RESULTS\n")
cat("=============================================\n\n")

dnn_results <- list(
  model   = dnn_final,
  history = history,
  cm      = cm,
  roc     = roc_obj,
  auc     = auc_val,
  metrics = data.frame(
    Model       = "Deep Neural Network",
    Accuracy    = accuracy,
    Precision   = precision,
    Recall      = recall,
    F1_Score    = f1,
    Specificity = specificity,
    AUC_ROC     = auc_val
  )
)

saveRDS(dnn_results, "dnn_results.rds")
save_model(dnn_results$model, "dnn_keras_model.keras")
cat(">> Results saved to: dnn_results.rds\n")
cat("   (Compatible with model_comparison.R)\n\n")

# Save metrics CSV for report table
write.csv(dnn_results$metrics, "dnn_metrics.csv", row.names = FALSE)
cat(">> Metrics saved to: dnn_metrics.csv\n\n")

# =============================================================================
# SUMMARY — COPY THESE VALUES INTO TABLES 2 AND 3 OF THE REPORT
# =============================================================================

cat("=============================================\n")
cat("DEEP NEURAL NETWORK — FINAL SUMMARY\n")
cat("(Copy into Tables 2 & 3 of the report)\n")
cat("=============================================\n")
cat(sprintf("Accuracy    : %.2f%%\n", accuracy))
cat(sprintf("Precision   : %.2f%%\n", precision))
cat(sprintf("Recall      : %.2f%%\n", recall))
cat(sprintf("F1 Score    : %.2f%%\n", f1))
cat(sprintf("Specificity : %.2f%%\n", specificity))
cat(sprintf("AUC-ROC     : %.4f\n",   auc_val))
cat("\nConfusion matrix (replace [run*] placeholders):\n")
cat(sprintf("  TN = %d\n", cm$table["No",  "No"]))
cat(sprintf("  FP = %d\n", cm$table["No",  "Yes"]))
cat(sprintf("  FN = %d\n", cm$table["Yes", "No"]))
cat(sprintf("  TP = %d\n", cm$table["Yes", "Yes"]))
cat("\nOutput files:\n")
cat("  plot_16_dnn_confusion_matrix.png\n")
cat("  plot_17_dnn_roc_curve.png\n")
cat("  plot_18_dnn_training_loss.png\n")
cat("  plot_19_dnn_training_auc.png\n")
cat("  plot_20_four_model_comparison.png\n")
cat("  dnn_results.rds\n")
cat("  dnn_metrics.csv\n")
cat("=============================================\n")