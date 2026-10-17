# =============================================================================
# MODEL 1: LOGISTIC REGRESSION
# Big Data Module - Bankruptcy Prediction Dataset
# =============================================================================
# Logistic Regression is used as the baseline model. It is a well-established
# statistical method for binary classification, widely used in financial
# distress prediction (Altman & Sabato, 2007). It models the probability
# of bankruptcy as a function of the selected financial features.
#
# Run AFTER: feature_selection.R and data_preparation.R
# =============================================================================

# --- Install & Load Packages -------------------------------------------------

packages <- c("dplyr", "caret", "pROC", "ggplot2", "gridExtra")

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

library(dplyr)
library(caret)
library(pROC)
library(ggplot2)
library(gridExtra)

# =============================================================================
# STEP 1: LOAD PREPARED DATA
# =============================================================================

cat("=============================================\n")
cat("MODEL 1: LOGISTIC REGRESSION\n")
cat("=============================================\n\n")

train <- readRDS("train_scaled.rds")
test  <- readRDS("test_scaled.rds")

cat("Training set:", nrow(train), "rows |",
    "Class distribution:", paste(table(train$Bankrupt), collapse = " / "), "\n")
cat("Test set    :", nrow(test),  "rows |",
    "Class distribution:", paste(table(test$Bankrupt),  collapse = " / "), "\n\n")

# =============================================================================
# STEP 2: TRAIN LOGISTIC REGRESSION MODEL
# =============================================================================
# We use 10-fold cross-validation during training to get a more reliable
# estimate of model performance and reduce overfitting.

cat("=============================================\n")
cat("STEP 2: TRAINING MODEL\n")
cat("=============================================\n")

set.seed(42)

# 10-fold cross-validation
cv_control <- trainControl(
  method          = "cv",
  number          = 10,
  classProbs      = TRUE,        # needed for AUC-ROC
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# Logistic regression via glm
# Note: caret requires factor levels to be valid R names
levels(train$Bankrupt) <- c("No", "Yes")
levels(test$Bankrupt)  <- c("No", "Yes")

lr_model <- train(
  Bankrupt ~ .,
  data      = train,
  method    = "glm",
  family    = "binomial",
  trControl = cv_control,
  metric    = "ROC"             # optimise for AUC-ROC given class imbalance
)

cat("Model training complete.\n\n")
cat("Cross-validation results:\n")
print(lr_model$results)

# =============================================================================
# STEP 3: PREDICTIONS ON TEST SET
# =============================================================================

cat("\n=============================================\n")
cat("STEP 3: PREDICTIONS ON TEST SET\n")
cat("=============================================\n")

# Class predictions
lr_pred_class <- predict(lr_model, newdata = test)

# Probability predictions (for ROC curve)
lr_pred_prob  <- predict(lr_model, newdata = test, type = "prob")[, "Yes"]

cat("Prediction distribution:\n")
print(table(lr_pred_class))

# =============================================================================
# STEP 4: MODEL EVALUATION
# =============================================================================

cat("\n=============================================\n")
cat("STEP 4: MODEL EVALUATION\n")
cat("=============================================\n")

# --- Confusion Matrix --------------------------------------------------------
cm <- confusionMatrix(
  data      = lr_pred_class,
  reference = test$Bankrupt,
  positive  = "Yes"
)

cat("Confusion Matrix:\n")
print(cm$table)
cat("\n")

# Extract key metrics
accuracy  <- round(cm$overall["Accuracy"] * 100, 2)
precision <- round(cm$byClass["Precision"] * 100, 2)
recall    <- round(cm$byClass["Recall"] * 100, 2)
f1        <- round(cm$byClass["F1"] * 100, 2)
specificity <- round(cm$byClass["Specificity"] * 100, 2)

cat(sprintf("Accuracy    : %.2f%%\n", accuracy))
cat(sprintf("Precision   : %.2f%%\n", precision))
cat(sprintf("Recall      : %.2f%%\n", recall))
cat(sprintf("F1 Score    : %.2f%%\n", f1))
cat(sprintf("Specificity : %.2f%%\n", specificity))

# --- AUC-ROC ----------------------------------------------------------------
roc_obj <- roc(
  response  = test$Bankrupt,
  predictor = lr_pred_prob,
  levels    = c("No", "Yes")
)
auc_val <- round(auc(roc_obj), 4)
cat(sprintf("AUC-ROC     : %.4f\n", auc_val))

# =============================================================================
# STEP 5: VISUALISATIONS
# =============================================================================

cat("\n=============================================\n")
cat("STEP 5: VISUALISATIONS\n")
cat("=============================================\n")

# --- Confusion Matrix Heatmap -----------------------------------------------
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Predicted", "Actual", "Count")

p_cm <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = Count), size = 8, fontface = "bold", colour = "white") +
  scale_fill_gradient(low = "#2980B9", high = "#1A252F") +
  labs(
    title    = "Logistic Regression — Confusion Matrix",
    subtitle = "Test set predictions",
    x = "Predicted Class", y = "Actual Class"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("plot_09_lr_confusion_matrix.png",
       plot = p_cm, width = 6, height = 5, dpi = 150)
cat(">> Saved: plot_09_lr_confusion_matrix.png\n")

# --- ROC Curve --------------------------------------------------------------
roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(colour = "#2980B9", linewidth = 1.2) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  annotate("text", x = 0.65, y = 0.15,
           label = paste0("AUC = ", auc_val),
           size = 5, colour = "#2980B9", fontface = "bold") +
  labs(
    title    = "Logistic Regression — ROC Curve",
    subtitle = "Test set performance",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal(base_size = 13)

ggsave("plot_10_lr_roc_curve.png",
       plot = p_roc, width = 6, height = 5, dpi = 150)
cat(">> Saved: plot_10_lr_roc_curve.png\n")

# --- Predicted Probability Distribution ------------------------------------
prob_df <- data.frame(
  Probability = lr_pred_prob,
  Actual      = test$Bankrupt
)

p_prob <- ggplot(prob_df, aes(x = Probability, fill = Actual)) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("No" = "#2ECC71", "Yes" = "#E74C3C"),
                    labels  = c("No" = "Not Bankrupt", "Yes" = "Bankrupt")) +
  labs(
    title    = "Logistic Regression — Predicted Probability Distribution",
    subtitle = "Distribution of predicted bankruptcy probabilities by actual class",
    x = "Predicted Probability of Bankruptcy",
    y = "Count", fill = ""
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("plot_11_lr_prob_distribution.png",
       plot = p_prob, width = 8, height = 5, dpi = 150)
cat(">> Saved: plot_11_lr_prob_distribution.png\n")

# =============================================================================
# STEP 6: SAVE RESULTS FOR MODEL COMPARISON
# =============================================================================

cat("\n=============================================\n")
cat("STEP 6: SAVE RESULTS\n")
cat("=============================================\n")

lr_results <- list(
  model      = lr_model,
  cm         = cm,
  roc        = roc_obj,
  auc        = auc_val,
  metrics    = data.frame(
    Model       = "Logistic Regression",
    Accuracy    = accuracy,
    Precision   = precision,
    Recall      = recall,
    F1_Score    = f1,
    Specificity = specificity,
    AUC_ROC     = auc_val
  )
)

saveRDS(lr_results, "lr_results.rds")
cat(">> Results saved to: lr_results.rds\n")
cat("   (Used in model comparison script)\n")

# =============================================================================
# SUMMARY
# =============================================================================

cat("\n=============================================\n")
cat("LOGISTIC REGRESSION — SUMMARY\n")
cat("=============================================\n")
cat(sprintf("Accuracy    : %.2f%%\n", accuracy))
cat(sprintf("Precision   : %.2f%%\n", precision))
cat(sprintf("Recall      : %.2f%%\n", recall))
cat(sprintf("F1 Score    : %.2f%%\n", f1))
cat(sprintf("Specificity : %.2f%%\n", specificity))
cat(sprintf("AUC-ROC     : %.4f\n",   auc_val))
cat("\nOutput files:\n")
cat("  plot_09_lr_confusion_matrix.png\n")
cat("  plot_10_lr_roc_curve.png\n")
cat("  plot_11_lr_prob_distribution.png\n")
cat("  lr_results.rds\n")
cat("=============================================\n")