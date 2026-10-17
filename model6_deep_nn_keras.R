# =============================================================================
# MODEL 6: DEEP NEURAL NETWORK (Multi-layer MLP via Keras / TensorFlow)
# Big Data Module - Bankruptcy Prediction Dataset
# =============================================================================
# A deeper MLP than model3_neural_network.R (which uses neuralnet with 16->8).
# This model uses Keras with 4 hidden layers, batch normalisation, dropout
# regularisation, early stopping, and class-weighted loss to address imbalance.
#
# Architecture:
#   Input (22) -> Dense(64, ReLU) -> BN -> Dropout(0.3)
#              -> Dense(32, ReLU) -> BN -> Dropout(0.3)
#              -> Dense(16, ReLU) -> Dropout(0.2)
#              -> Dense(8,  ReLU) -> Dropout(0.2)
#              -> Dense(1, Sigmoid)  [binary output]
#
# FIRST-TIME SETUP (run once in console before sourcing this script):
#   install.packages("keras")
#   keras::install_keras()   # installs the TensorFlow / Python backend
#
# Run AFTER: feature_selection.R and data_preparation.R
# =============================================================================

# --- Install & Load Packages -------------------------------------------------

packages <- c("dplyr", "keras", "pROC", "ggplot2", "caret")

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

library(dplyr)
library(keras)
library(pROC)
library(ggplot2)
library(caret)

# =============================================================================
# STEP 1: LOAD PREPARED DATA
# =============================================================================

cat("=============================================\n")
cat("MODEL 6: DEEP NEURAL NETWORK (KERAS)\n")
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

# =============================================================================
# STEP 2: PREPARE MATRICES FOR KERAS
# =============================================================================
# Keras requires numeric matrices — no factor columns.
# Target is encoded as 0 (No) / 1 (Yes).

cat("=============================================\n")
cat("STEP 2: PREPARING DATA FOR KERAS\n")
cat("=============================================\n")

feature_cols <- names(train)[names(train) != "Bankrupt"]
n_features   <- length(feature_cols)

train_x <- as.matrix(train[, feature_cols])
train_y <- as.numeric(train$Bankrupt == "Yes")   # 1 = bankrupt, 0 = not

test_x  <- as.matrix(test[, feature_cols])
test_y  <- as.numeric(test$Bankrupt == "Yes")

cat("Input features  :", n_features, "\n")
cat("Training samples:", nrow(train_x),
    "| Bankrupt:", sum(train_y),
    "| Not bankrupt:", sum(train_y == 0), "\n")
cat("Test samples    :", nrow(test_x),
    "| Bankrupt:", sum(test_y),
    "| Not bankrupt:", sum(test_y == 0), "\n\n")

# --- Class weights (down-weight majority to handle any residual imbalance) ---
n_no    <- sum(train_y == 0)
n_yes   <- sum(train_y == 1)
total   <- n_no + n_yes

class_weight <- list(
  "0" = total / (2 * n_no),   # weight for not-bankrupt
  "1" = total / (2 * n_yes)   # weight for bankrupt
)

cat(sprintf("Class weights  — 0 (No): %.4f | 1 (Yes): %.4f\n\n",
            class_weight[["0"]], class_weight[["1"]]))

# =============================================================================
# STEP 3: BUILD THE DEEP NEURAL NETWORK
# =============================================================================
# 4 hidden layers with progressive compression: 64 -> 32 -> 16 -> 8
# Batch normalisation stabilises training; dropout prevents overfitting.

cat("=============================================\n")
cat("STEP 3: BUILDING MODEL ARCHITECTURE\n")
cat("=============================================\n")

set.seed(42)
tensorflow::set_random_seed(42)

model <- keras_model_sequential(name = "bankruptcy_dnn") %>%

  # --- Hidden Layer 1 : 64 units ---
  layer_dense(units = 64, activation = "relu",
              input_shape = n_features, name = "dense_1") %>%
  layer_batch_normalization(name = "bn_1") %>%
  layer_dropout(rate = 0.3, name = "drop_1") %>%

  # --- Hidden Layer 2 : 32 units ---
  layer_dense(units = 32, activation = "relu", name = "dense_2") %>%
  layer_batch_normalization(name = "bn_2") %>%
  layer_dropout(rate = 0.3, name = "drop_2") %>%

  # --- Hidden Layer 3 : 16 units ---
  layer_dense(units = 16, activation = "relu", name = "dense_3") %>%
  layer_dropout(rate = 0.2, name = "drop_3") %>%

  # --- Hidden Layer 4 : 8 units ---
  layer_dense(units = 8, activation = "relu", name = "dense_4") %>%
  layer_dropout(rate = 0.2, name = "drop_4") %>%

  # --- Output Layer : sigmoid for binary probability ---
  layer_dense(units = 1, activation = "sigmoid", name = "output")

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss      = "binary_crossentropy",
  metrics   = list("accuracy", metric_auc(name = "auc"))
)

cat("\nModel architecture:\n")
summary(model)

# =============================================================================
# STEP 4: TRAIN THE MODEL
# =============================================================================
# Early stopping monitors val_auc and restores the best weights automatically.
# ReduceLROnPlateau halves the learning rate if val_loss stalls for 5 epochs.

cat("\n=============================================\n")
cat("STEP 4: TRAINING MODEL\n")
cat("=============================================\n")
cat("Up to 100 epochs | batch size 256 | 15%% validation split\n")
cat("Early stopping patience: 10 epochs (monitoring val_auc)\n\n")

callbacks_list <- list(
  callback_early_stopping(
    monitor              = "val_auc",
    patience             = 10,
    mode                 = "max",
    restore_best_weights = TRUE,
    verbose              = 1
  ),
  callback_reduce_lr_on_plateau(
    monitor  = "val_loss",
    factor   = 0.5,
    patience = 5,
    min_lr   = 1e-6,
    verbose  = 1
  )
)

set.seed(42)
history <- model %>% fit(
  x                = train_x,
  y                = train_y,
  epochs           = 100,
  batch_size       = 256,
  validation_split = 0.15,
  class_weight     = class_weight,
  callbacks        = callbacks_list,
  verbose          = 1
)

cat("\nTraining complete.\n")
cat("Epochs run         :", length(history$metrics$loss), "\n")
cat("Final train loss   :", round(tail(history$metrics$loss, 1), 4), "\n")
cat("Final val loss     :", round(tail(history$metrics$val_loss, 1), 4), "\n")
cat("Final train AUC    :", round(tail(history$metrics$auc, 1), 4), "\n")
cat("Final val AUC      :", round(tail(history$metrics$val_auc, 1), 4), "\n\n")

# =============================================================================
# STEP 5: PREDICTIONS ON TEST SET
# =============================================================================

cat("=============================================\n")
cat("STEP 5: PREDICTIONS ON TEST SET\n")
cat("=============================================\n")

# Raw predicted probabilities
dnn_pred_prob  <- as.vector(model %>% predict(test_x))

# Class predictions at 0.5 threshold
dnn_pred_class <- factor(
  ifelse(dnn_pred_prob >= 0.5, "Yes", "No"),
  levels = c("No", "Yes")
)

cat("Prediction distribution:\n")
print(table(dnn_pred_class))

# =============================================================================
# STEP 6: MODEL EVALUATION
# =============================================================================

cat("\n=============================================\n")
cat("STEP 6: MODEL EVALUATION\n")
cat("=============================================\n")

test_factor <- factor(ifelse(test_y == 1, "Yes", "No"), levels = c("No", "Yes"))

# --- Confusion Matrix --------------------------------------------------------
cm <- confusionMatrix(
  data      = dnn_pred_class,
  reference = test_factor,
  positive  = "Yes"
)

cat("Confusion Matrix:\n")
print(cm$table)
cat("\n")

# Extract key metrics
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

# --- AUC-ROC ----------------------------------------------------------------
roc_obj <- roc(
  response  = test_y,
  predictor = dnn_pred_prob
)
auc_val <- round(auc(roc_obj), 4)
cat(sprintf("AUC-ROC     : %.4f\n", auc_val))

# =============================================================================
# STEP 7: VISUALISATIONS
# =============================================================================

cat("\n=============================================\n")
cat("STEP 7: VISUALISATIONS\n")
cat("=============================================\n")

DNN_COLOUR <- "#C0392B"   # crimson — distinct from LR/RF/NN/SVM/AdaBoost

# --- Confusion Matrix Heatmap -----------------------------------------------
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Predicted", "Actual", "Count")

p_cm <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = Count), size = 8, fontface = "bold", colour = "white") +
  scale_fill_gradient(low = DNN_COLOUR, high = "#641E16") +
  labs(
    title    = "Deep Neural Network (Keras) — Confusion Matrix",
    subtitle = "Test set predictions",
    x = "Predicted Class", y = "Actual Class"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("plot_32_dnn_confusion_matrix.png",
       plot = p_cm, width = 6, height = 5, dpi = 150)
cat(">> Saved: plot_32_dnn_confusion_matrix.png\n")

# --- ROC Curve --------------------------------------------------------------
roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(colour = DNN_COLOUR, linewidth = 1.2) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  annotate("text", x = 0.65, y = 0.15,
           label = paste0("AUC = ", auc_val),
           size = 5, colour = DNN_COLOUR, fontface = "bold") +
  labs(
    title    = "Deep Neural Network (Keras) — ROC Curve",
    subtitle = "Test set performance",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal(base_size = 13)

ggsave("plot_33_dnn_roc_curve.png",
       plot = p_roc, width = 6, height = 5, dpi = 150)
cat(">> Saved: plot_33_dnn_roc_curve.png\n")

# --- Predicted Probability Distribution -------------------------------------
prob_df <- data.frame(
  Probability = dnn_pred_prob,
  Actual      = factor(ifelse(test_y == 1, "Bankrupt", "Not Bankrupt"),
                       levels = c("Not Bankrupt", "Bankrupt"))
)

p_prob <- ggplot(prob_df, aes(x = Probability, fill = Actual)) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("Not Bankrupt" = "#2ECC71",
                               "Bankrupt"     = "#E74C3C")) +
  labs(
    title    = "Deep Neural Network (Keras) — Predicted Probability Distribution",
    subtitle = "Distribution of predicted bankruptcy probabilities by actual class",
    x = "Predicted Probability of Bankruptcy",
    y = "Count", fill = ""
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("plot_34_dnn_prob_distribution.png",
       plot = p_prob, width = 8, height = 5, dpi = 150)
cat(">> Saved: plot_34_dnn_prob_distribution.png\n")

# --- Training History Plot --------------------------------------------------
# Loss and AUC over epochs for both train and validation sets
epochs_run <- length(history$metrics$loss)

history_df <- data.frame(
  Epoch    = rep(1:epochs_run, 4),
  Value    = c(history$metrics$loss,
               history$metrics$val_loss,
               history$metrics$auc,
               history$metrics$val_auc),
  Metric   = rep(c("Train Loss", "Val Loss", "Train AUC", "Val AUC"),
                 each = epochs_run)
)
history_df$Type  <- ifelse(grepl("Loss", history_df$Metric), "Loss", "AUC")
history_df$Split <- ifelse(grepl("Train", history_df$Metric), "Train", "Validation")

p_hist <- ggplot(history_df, aes(x = Epoch, y = Value,
                                  colour = Split, linetype = Split)) +
  geom_line(linewidth = 1) +
  facet_wrap(~Type, scales = "free_y") +
  scale_colour_manual(values = c("Train" = DNN_COLOUR, "Validation" = "#566573")) +
  scale_linetype_manual(values = c("Train" = "solid", "Validation" = "dashed")) +
  labs(
    title    = "Deep Neural Network — Training History",
    subtitle = paste0("Stopped at epoch ", epochs_run, " (early stopping on val_auc)"),
    x = "Epoch", y = "Value", colour = "", linetype = ""
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        strip.text = element_text(face = "bold", size = 11))

ggsave("plot_35_dnn_training_history.png",
       plot = p_hist, width = 10, height = 5, dpi = 150)
cat(">> Saved: plot_35_dnn_training_history.png\n")

# =============================================================================
# STEP 8: SAVE RESULTS FOR MODEL COMPARISON
# =============================================================================

cat("\n=============================================\n")
cat("STEP 8: SAVE RESULTS\n")
cat("=============================================\n")

dnn_results <- list(
  model   = model,
  history = history,
  cm      = cm,
  roc     = roc_obj,
  auc     = auc_val,
  metrics = data.frame(
    Model       = "Deep NN (Keras)",
    Accuracy    = accuracy,
    Precision   = precision,
    Recall      = recall,
    F1_Score    = f1,
    Specificity = specificity,
    AUC_ROC     = auc_val
  )
)

saveRDS(dnn_results, "dnn_results.rds")
cat(">> Results saved to: dnn_results.rds\n")
cat("   (Can be loaded in model comparison script alongside other _results.rds files)\n")

# =============================================================================
# SUMMARY
# =============================================================================

cat("\n=============================================\n")
cat("DEEP NEURAL NETWORK (KERAS) — SUMMARY\n")
cat("=============================================\n")
cat("Architecture: Input -> 64 -> 32 -> 16 -> 8 -> Output\n")
cat("Activations : ReLU (hidden) | Sigmoid (output)\n")
cat("Regulariser : Batch Norm + Dropout (0.3 / 0.3 / 0.2 / 0.2)\n")
cat("Optimizer   : Adam (lr=0.001, reduce on plateau)\n")
cat("Loss        : Binary cross-entropy (class-weighted)\n")
cat(sprintf("Epochs run  : %d (early stopping)\n", epochs_run))
cat(sprintf("Accuracy    : %.2f%%\n", accuracy))
cat(sprintf("Precision   : %.2f%%\n", precision))
cat(sprintf("Recall      : %.2f%%\n", recall))
cat(sprintf("F1 Score    : %.2f%%\n", f1))
cat(sprintf("Specificity : %.2f%%\n", specificity))
cat(sprintf("AUC-ROC     : %.4f\n",   auc_val))
cat("\nOutput files:\n")
cat("  plot_32_dnn_confusion_matrix.png\n")
cat("  plot_33_dnn_roc_curve.png\n")
cat("  plot_34_dnn_prob_distribution.png\n")
cat("  plot_35_dnn_training_history.png\n")
cat("  dnn_results.rds\n")
cat("=============================================\n")
