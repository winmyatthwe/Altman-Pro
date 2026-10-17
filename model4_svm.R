# =============================================================================
# MODEL 4: SUPPORT VECTOR MACHINE (SVM — RBF Kernel)
# Big Data Module - Bankruptcy Prediction Dataset
# =============================================================================
# Support Vector Machine finds the optimal hyperplane that maximises the margin
# between classes. The Radial Basis Function (RBF) kernel maps inputs into a
# higher-dimensional space to capture non-linear decision boundaries — well
# suited to financial distress prediction (Min & Lee, 2005; Huang et al., 2004).
#
# Key hyperparameters tuned:
#   C     — regularisation cost (margin width vs. misclassification trade-off)
#   sigma — RBF bandwidth (localness of each training point's influence)
#
# Run AFTER: feature_selection.R and data_preparation.R
# =============================================================================

# --- Install & Load Packages -------------------------------------------------

packages <- c("dplyr", "caret", "kernlab", "pROC", "ggplot2")

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

library(dplyr)
library(caret)
library(kernlab)
library(pROC)
library(ggplot2)

# =============================================================================
# STEP 1: LOAD PREPARED DATA
# =============================================================================

cat("=============================================\n")
cat("MODEL 4: SUPPORT VECTOR MACHINE (SVM-RBF)\n")
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
# STEP 2: HYPERPARAMETER TUNING (C and sigma)
# =============================================================================
# SVM with RBF kernel has two hyperparameters:
#   C     — higher C = tighter fit to training data (risk of overfitting)
#   sigma — smaller sigma = smoother decision boundary (more generalisation)
#
# We use 5-fold cross-validation (rather than 10-fold) to keep training time
# manageable. Optimise for AUC-ROC given the class imbalance in the data.

cat("=============================================\n")
cat("STEP 2: HYPERPARAMETER TUNING (C, sigma)\n")
cat("=============================================\n")
cat("Using 5-fold cross-validation...\n")
cat("NOTE: SVM tuning on large datasets can take several minutes.\n\n")

set.seed(42)

cv_control <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,        # needed for probability estimates and AUC
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# Grid covering a range around the rule-of-thumb defaults
tune_grid <- expand.grid(
  sigma = c(0.01, 0.05, 0.10),
  C     = c(0.5,  1,    5,   10)
)

cat("Tuning grid (", nrow(tune_grid), "combinations x 5 folds =",
    nrow(tune_grid) * 5, "model fits):\n")
print(tune_grid)
cat("\n")

svm_tuned <- caret::train(
  Bankrupt ~ .,
  data      = train,
  method    = "svmRadial",
  trControl = cv_control,
  tuneGrid  = tune_grid,
  metric    = "ROC"
)

cat("\nTuning complete.\n")
cat("Cross-validation results:\n")
print(svm_tuned$results[, c("sigma", "C", "ROC", "Sens", "Spec")])

best_sigma <- svm_tuned$bestTune$sigma
best_C     <- svm_tuned$bestTune$C
cat(sprintf("\nBest sigma : %.4f\n", best_sigma))
cat(sprintf("Best C     : %.2f\n",  best_C))

# =============================================================================
# STEP 3: TRAIN FINAL MODEL WITH BEST HYPERPARAMETERS
# =============================================================================
# Refit on the full training set using the best (sigma, C) found above.
# method = "none" skips cross-validation for this final fit.

cat("\n=============================================\n")
cat("STEP 3: TRAINING FINAL MODEL\n")
cat("=============================================\n")
cat(sprintf("Training SVM-RBF  sigma = %.4f | C = %.2f\n\n", best_sigma, best_C))

set.seed(42)

svm_final <- caret::train(
  Bankrupt ~ .,
  data      = train,
  method    = "svmRadial",
  trControl = trainControl(
    method     = "none",
    classProbs = TRUE
  ),
  tuneGrid  = data.frame(sigma = best_sigma, C = best_C)
)

cat("Model training complete.\n\n")
print(svm_final)

# =============================================================================
# STEP 4: PREDICTIONS ON TEST SET
# =============================================================================

cat("\n=============================================\n")
cat("STEP 4: PREDICTIONS ON TEST SET\n")
cat("=============================================\n")

# Class predictions
svm_pred_class <- predict(svm_final, newdata = test)

# Probability predictions (required for AUC-ROC)
svm_pred_prob  <- predict(svm_final, newdata = test, type = "prob")[, "Yes"]

cat("Prediction distribution:\n")
print(table(svm_pred_class))

# =============================================================================
# STEP 5: MODEL EVALUATION
# =============================================================================

cat("\n=============================================\n")
cat("STEP 5: MODEL EVALUATION\n")
cat("=============================================\n")

# --- Confusion Matrix --------------------------------------------------------
cm <- confusionMatrix(
  data      = svm_pred_class,
  reference = test$Bankrupt,
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
  response  = test$Bankrupt,
  predictor = svm_pred_prob,
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

SVM_COLOUR <- "#E67E22"   # orange — distinct from LR (blue), RF (green), NN (purple)

# --- Confusion Matrix Heatmap -----------------------------------------------
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Predicted", "Actual", "Count")

p_cm <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = Count), size = 8, fontface = "bold", colour = "white") +
  scale_fill_gradient(low = SVM_COLOUR, high = "#784212") +
  labs(
    title    = "SVM (RBF) — Confusion Matrix",
    subtitle = "Test set predictions",
    x = "Predicted Class", y = "Actual Class"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("plot_24_svm_confusion_matrix.png",
       plot = p_cm, width = 6, height = 5, dpi = 150)
cat(">> Saved: plot_24_svm_confusion_matrix.png\n")

# --- ROC Curve --------------------------------------------------------------
roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(colour = SVM_COLOUR, linewidth = 1.2) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  annotate("text", x = 0.65, y = 0.15,
           label = paste0("AUC = ", auc_val),
           size = 5, colour = SVM_COLOUR, fontface = "bold") +
  labs(
    title    = "SVM (RBF) — ROC Curve",
    subtitle = "Test set performance",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal(base_size = 13)

ggsave("plot_25_svm_roc_curve.png",
       plot = p_roc, width = 6, height = 5, dpi = 150)
cat(">> Saved: plot_25_svm_roc_curve.png\n")

# --- Predicted Probability Distribution -------------------------------------
prob_df <- data.frame(
  Probability = svm_pred_prob,
  Actual      = test$Bankrupt
)

p_prob <- ggplot(prob_df, aes(x = Probability, fill = Actual)) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("No" = "#2ECC71", "Yes" = "#E74C3C"),
                    labels  = c("No" = "Not Bankrupt", "Yes" = "Bankrupt")) +
  labs(
    title    = "SVM (RBF) — Predicted Probability Distribution",
    subtitle = "Distribution of predicted bankruptcy probabilities by actual class",
    x = "Predicted Probability of Bankruptcy",
    y = "Count", fill = ""
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("plot_26_svm_prob_distribution.png",
       plot = p_prob, width = 8, height = 5, dpi = 150)
cat(">> Saved: plot_26_svm_prob_distribution.png\n")

# --- Hyperparameter Tuning Surface ------------------------------------------
# Shows how ROC varies across the C x sigma grid
tune_results <- svm_tuned$results

p_tune <- ggplot(tune_results,
                 aes(x = factor(C), y = ROC,
                     colour = factor(sigma), group = factor(sigma))) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_colour_brewer(palette = "Set1", name = "sigma") +
  labs(
    title    = "SVM — Cross-Validation Tuning Results",
    subtitle = "5-fold CV AUC-ROC across C and sigma grid",
    x = "Cost (C)", y = "CV AUC-ROC"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("plot_27_svm_tuning.png",
       plot = p_tune, width = 7, height = 5, dpi = 150)
cat(">> Saved: plot_27_svm_tuning.png\n")

# =============================================================================
# STEP 7: SAVE RESULTS FOR MODEL COMPARISON
# =============================================================================

cat("\n=============================================\n")
cat("STEP 7: SAVE RESULTS\n")
cat("=============================================\n")

svm_results <- list(
  model   = svm_final,
  cm      = cm,
  roc     = roc_obj,
  auc     = auc_val,
  metrics = data.frame(
    Model       = "SVM (RBF)",
    Accuracy    = accuracy,
    Precision   = precision,
    Recall      = recall,
    F1_Score    = f1,
    Specificity = specificity,
    AUC_ROC     = auc_val
  )
)

saveRDS(svm_results, "svm_results.rds")
cat(">> Results saved to: svm_results.rds\n")
cat("   (Can be loaded in model comparison script alongside lr/rf/nn_results.rds)\n")

# =============================================================================
# SUMMARY
# =============================================================================

cat("\n=============================================\n")
cat("SVM (RBF) — SUMMARY\n")
cat("=============================================\n")
cat(sprintf("Kernel      : Radial Basis Function (RBF)\n"))
cat(sprintf("Best sigma  : %.4f\n",  best_sigma))
cat(sprintf("Best C      : %.2f\n",  best_C))
cat(sprintf("Accuracy    : %.2f%%\n", accuracy))
cat(sprintf("Precision   : %.2f%%\n", precision))
cat(sprintf("Recall      : %.2f%%\n", recall))
cat(sprintf("F1 Score    : %.2f%%\n", f1))
cat(sprintf("Specificity : %.2f%%\n", specificity))
cat(sprintf("AUC-ROC     : %.4f\n",   auc_val))
cat("\nOutput files:\n")
cat("  plot_24_svm_confusion_matrix.png\n")
cat("  plot_25_svm_roc_curve.png\n")
cat("  plot_26_svm_prob_distribution.png\n")
cat("  plot_27_svm_tuning.png\n")
cat("  svm_results.rds\n")
cat("=============================================\n")




svm_results <- readRDS("svm_results.rds")
train <- readRDS("train_scaled.rds")
# See exactly what columns the SVM was trained on:
print(svm_results$model$finalModel@terms)
# Or simply:
print(names(train))