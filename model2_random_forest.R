# =============================================================================
# MODEL 2: RANDOM FOREST
# Big Data Module - Bankruptcy Prediction Dataset
# =============================================================================
# Random Forest is an ensemble learning method that builds multiple decision
# trees and combines their predictions. It is robust to outliers, handles
# non-linear relationships, and provides feature importance scores — making
# it well suited to financial distress prediction (Breiman, 2001).
#
# Run AFTER: feature_selection.R and data_preparation.R
# =============================================================================

# --- Install & Load Packages -------------------------------------------------

packages <- c("dplyr", "caret", "randomForest", "pROC", "ggplot2")

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

library(dplyr)
library(caret)
library(randomForest)
library(pROC)
library(ggplot2)

# =============================================================================
# STEP 1: LOAD PREPARED DATA
# =============================================================================

cat("=============================================\n")
cat("MODEL 2: RANDOM FOREST\n")
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
# STEP 2: HYPERPARAMETER TUNING
# =============================================================================
# Random Forest has one key hyperparameter to tune:
#   mtry — the number of features randomly sampled at each tree split
# We use 5-fold cross-validation to find the optimal mtry value.
# (5-fold rather than 10-fold to keep computation time reasonable)

cat("=============================================\n")
cat("STEP 2: HYPERPARAMETER TUNING (mtry)\n")
cat("=============================================\n")
cat("This may take several minutes...\n\n")

set.seed(42)

cv_control <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# Define mtry values to test
# Rule of thumb: sqrt(number of features) as starting point
n_features  <- ncol(train) - 1
mtry_default <- floor(sqrt(n_features))

tune_grid <- expand.grid(
  mtry = c(2, mtry_default, mtry_default * 2, n_features / 2)
)

cat("Number of features:", n_features, "\n")
cat("Default mtry (sqrt of features):", mtry_default, "\n")
cat("Testing mtry values:", paste(tune_grid$mtry, collapse = ", "), "\n\n")

rf_tuned <- train(
  Bankrupt ~ .,
  data      = train,
  method    = "rf",
  trControl = cv_control,
  tuneGrid  = tune_grid,
  metric    = "ROC",
  ntree     = 300
)

cat("Tuning results:\n")
print(rf_tuned$results[, c("mtry", "ROC", "Sens", "Spec")])

best_mtry <- rf_tuned$bestTune$mtry
cat(sprintf("\nBest mtry: %d\n", best_mtry))

# =============================================================================
# STEP 3: TRAIN FINAL MODEL WITH BEST HYPERPARAMETERS
# =============================================================================

cat("\n=============================================\n")
cat("STEP 3: TRAINING FINAL MODEL\n")
cat("=============================================\n")
cat(sprintf("Training Random Forest with mtry = %d, ntree = 500\n\n", best_mtry))

set.seed(42)

rf_final <- randomForest(
  Bankrupt ~ .,
  data       = train,
  ntree      = 500,
  mtry       = best_mtry,
  importance = TRUE
)

cat("Model training complete.\n\n")
print(rf_final)

# =============================================================================
# STEP 4: PREDICTIONS ON TEST SET
# =============================================================================

cat("\n=============================================\n")
cat("STEP 4: PREDICTIONS ON TEST SET\n")
cat("=============================================\n")

# Class predictions
rf_pred_class <- predict(rf_final, newdata = test, type = "class")

# Probability predictions (for ROC curve)
rf_pred_prob  <- predict(rf_final, newdata = test, type = "prob")[, "Yes"]

cat("Prediction distribution:\n")
print(table(rf_pred_class))

# =============================================================================
# STEP 5: MODEL EVALUATION
# =============================================================================

cat("\n=============================================\n")
cat("STEP 5: MODEL EVALUATION\n")
cat("=============================================\n")

# --- Confusion Matrix --------------------------------------------------------
cm <- confusionMatrix(
  data      = rf_pred_class,
  reference = test$Bankrupt,
  positive  = "Yes"
)

cat("Confusion Matrix:\n")
print(cm$table)
cat("\n")

# Extract key metrics
accuracy    <- round(cm$overall["Accuracy"] * 100, 2)
precision   <- round(cm$byClass["Precision"] * 100, 2)
recall      <- round(cm$byClass["Recall"] * 100, 2)
f1          <- round(cm$byClass["F1"] * 100, 2)
specificity <- round(cm$byClass["Specificity"] * 100, 2)

cat(sprintf("Accuracy    : %.2f%%\n", accuracy))
cat(sprintf("Precision   : %.2f%%\n", precision))
cat(sprintf("Recall      : %.2f%%\n", recall))
cat(sprintf("F1 Score    : %.2f%%\n", f1))
cat(sprintf("Specificity : %.2f%%\n", specificity))

# --- AUC-ROC ----------------------------------------------------------------
roc_obj <- roc(
  response  = test$Bankrupt,
  predictor = rf_pred_prob,
  levels    = c("No", "Yes")
)
auc_val <- round(auc(roc_obj), 4)
cat(sprintf("AUC-ROC     : %.4f\n", auc_val))

# =============================================================================
# STEP 6: VISUALISATIONS
# =============================================================================

cat("\n=============================================\n")
cat("STEP 6: VISUALISATIONS\n")
cat("=============================================\n")

# --- Confusion Matrix Heatmap -----------------------------------------------
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Predicted", "Actual", "Count")

p_cm <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = Count), size = 8, fontface = "bold", colour = "white") +
  scale_fill_gradient(low = "#27AE60", high = "#1A5632") +
  labs(
    title    = "Random Forest — Confusion Matrix",
    subtitle = "Test set predictions",
    x = "Predicted Class", y = "Actual Class"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("plot_12_rf_confusion_matrix.png",
       plot = p_cm, width = 6, height = 5, dpi = 150)
cat(">> Saved: plot_12_rf_confusion_matrix.png\n")

# --- ROC Curve --------------------------------------------------------------
roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(colour = "#27AE60", linewidth = 1.2) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  annotate("text", x = 0.65, y = 0.15,
           label = paste0("AUC = ", auc_val),
           size = 5, colour = "#27AE60", fontface = "bold") +
  labs(
    title    = "Random Forest — ROC Curve",
    subtitle = "Test set performance",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal(base_size = 13)

ggsave("plot_13_rf_roc_curve.png",
       plot = p_roc, width = 6, height = 5, dpi = 150)
cat(">> Saved: plot_13_rf_roc_curve.png\n")

# --- Feature Importance Plot ------------------------------------------------
importance_df <- data.frame(
  Feature          = rownames(importance(rf_final)),
  MeanDecreaseGini = importance(rf_final)[, "MeanDecreaseGini"]
) %>% arrange(desc(MeanDecreaseGini))

p_imp <- ggplot(importance_df,
                aes(x = reorder(Feature, MeanDecreaseGini),
                    y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "#27AE60") +
  geom_text(aes(label = round(MeanDecreaseGini, 1)),
            hjust = -0.1, size = 3) +
  coord_flip() +
  ylim(0, max(importance_df$MeanDecreaseGini) * 1.12) +
  labs(
    title    = "Random Forest — Feature Importance",
    subtitle = "Final model: ranked by Mean Decrease in Gini Impurity",
    x = "Feature", y = "Mean Decrease Gini"
  ) +
  theme_minimal(base_size = 10)

ggsave("plot_14_rf_feature_importance.png",
       plot = p_imp, width = 12, height = 8, dpi = 150)
cat(">> Saved: plot_14_rf_feature_importance.png\n")

# --- OOB Error Rate Plot ----------------------------------------------------
# Out-of-bag error shows how model error changes as more trees are added
oob_df <- data.frame(
  Trees    = 1:rf_final$ntree,
  OOB      = rf_final$err.rate[, "OOB"],
  No       = rf_final$err.rate[, "No"],
  Yes      = rf_final$err.rate[, "Yes"]
) %>%
  tidyr::pivot_longer(-Trees, names_to = "Class", values_to = "Error")

p_oob <- ggplot(oob_df, aes(x = Trees, y = Error, colour = Class)) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = c("OOB" = "black",
                                 "No"  = "#2ECC71",
                                 "Yes" = "#E74C3C"),
                      labels  = c("OOB" = "Overall OOB Error",
                                  "No"  = "Not Bankrupt Error",
                                  "Yes" = "Bankrupt Error")) +
  labs(
    title    = "Random Forest — Out-of-Bag Error Rate",
    subtitle = "Error stabilisation across number of trees",
    x = "Number of Trees", y = "OOB Error Rate", colour = ""
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("plot_15_rf_oob_error.png",
       plot = p_oob, width = 8, height = 5, dpi = 150)
cat(">> Saved: plot_15_rf_oob_error.png\n")

# =============================================================================
# STEP 7: SAVE RESULTS FOR MODEL COMPARISON
# =============================================================================

cat("\n=============================================\n")
cat("STEP 7: SAVE RESULTS\n")
cat("=============================================\n")

rf_results <- list(
  model   = rf_final,
  cm      = cm,
  roc     = roc_obj,
  auc     = auc_val,
  metrics = data.frame(
    Model       = "Random Forest",
    Accuracy    = accuracy,
    Precision   = precision,
    Recall      = recall,
    F1_Score    = f1,
    Specificity = specificity,
    AUC_ROC     = auc_val
  )
)

saveRDS(rf_results, "rf_results.rds")
cat(">> Results saved to: rf_results.rds\n")
cat("   (Used in model comparison script)\n")

# =============================================================================
# SUMMARY
# =============================================================================

cat("\n=============================================\n")
cat("RANDOM FOREST — SUMMARY\n")
cat("=============================================\n")
cat(sprintf("Best mtry   : %d\n",   best_mtry))
cat(sprintf("Num trees   : 500\n"))
cat(sprintf("Accuracy    : %.2f%%\n", accuracy))
cat(sprintf("Precision   : %.2f%%\n", precision))
cat(sprintf("Recall      : %.2f%%\n", recall))
cat(sprintf("F1 Score    : %.2f%%\n", f1))
cat(sprintf("Specificity : %.2f%%\n", specificity))
cat(sprintf("AUC-ROC     : %.4f\n",   auc_val))
cat("\nOutput files:\n")
cat("  plot_12_rf_confusion_matrix.png\n")
cat("  plot_13_rf_roc_curve.png\n")
cat("  plot_14_rf_feature_importance.png\n")
cat("  plot_15_rf_oob_error.png\n")
cat("  rf_results.rds\n")
cat("=============================================\n")