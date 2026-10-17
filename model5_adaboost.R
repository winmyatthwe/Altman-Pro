# =============================================================================
# MODEL 5: ADABOOST (Adaptive Boosting)
# Big Data Module - Bankruptcy Prediction Dataset
# =============================================================================
# AdaBoost is an ensemble boosting method that sequentially trains weak
# classifiers (shallow decision trees), each focusing on the examples
# misclassified by the previous one. It combines them into a strong classifier
# by weighted majority vote — well suited to imbalanced financial data
# (Freund & Schapire, 1997; Alfaro et al., 2008).
#
# Key hyperparameters tuned:
#   iter     — number of boosting rounds (more = stronger but slower)
#   maxdepth — depth of each base decision tree (1 = aclassic decision stump)
#   nu       — learning rate / shrinkage (smaller = more conservative updates)
#
# Run AFTER: feature_selection.R and data_preparation.R
# =============================================================================

# --- Install & Load Packages -------------------------------------------------

packages <- c("dplyr", "caret", "ada", "pROC", "ggplot2")

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

library(dplyr)
library(caret)
library(ada)
library(pROC)
library(ggplot2)

# =============================================================================
# STEP 1: LOAD PREPARED DATA
# =============================================================================

cat("=============================================\n")
cat("MODEL 5: ADABOOST\n")
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
# STEP 2: HYPERPARAMETER TUNING (iter, maxdepth, nu)
# =============================================================================
# AdaBoost has three main hyperparameters:
#   iter     — boosting rounds: more trees capture more complexity but risk overfitting
#   maxdepth — base learner depth: 1 = decision stump (classic AdaBoost.M1)
#   nu       — shrinkage: smaller values require more rounds but generalise better
#
# We use 5-fold cross-validation and optimise for AUC-ROC given class imbalance.

cat("=============================================\n")
cat("STEP 2: HYPERPARAMETER TUNING (iter, maxdepth, nu)\n")
cat("=============================================\n")
cat("Using 5-fold cross-validation...\n")
cat("NOTE: AdaBoost tuning may take several minutes.\n\n")

set.seed(42)

cv_control <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,        # needed for probability estimates and AUC
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# Grid: 3 iter x 2 maxdepth x 2 nu = 12 combinations x 5 folds = 60 model fits
tune_grid <- expand.grid(
  iter     = c(50, 100, 200),
  maxdepth = c(1, 2),
  nu       = c(0.1, 1)
)

cat("Tuning grid (", nrow(tune_grid), "combinations x 5 folds =",
    nrow(tune_grid) * 5, "model fits):\n")
print(tune_grid)
cat("\n")

ada_tuned <- caret::train(
  Bankrupt ~ .,
  data      = train,
  method    = "ada",
  trControl = cv_control,
  tuneGrid  = tune_grid,
  metric    = "ROC"
)

cat("\nTuning complete.\n")
cat("Cross-validation results:\n")
print(ada_tuned$results[, c("iter", "maxdepth", "nu", "ROC", "Sens", "Spec")])

best_iter     <- ada_tuned$bestTune$iter
best_maxdepth <- ada_tuned$bestTune$maxdepth
best_nu       <- ada_tuned$bestTune$nu

cat(sprintf("\nBest iter     : %d\n",   best_iter))
cat(sprintf("Best maxdepth : %d\n",   best_maxdepth))
cat(sprintf("Best nu       : %.2f\n", best_nu))

# =============================================================================
# STEP 3: TRAIN FINAL MODEL WITH BEST HYPERPARAMETERS
# =============================================================================
# Refit on the full training set using the best hyperparameters found above.

cat("\n=============================================\n")
cat("STEP 3: TRAINING FINAL MODEL\n")
cat("=============================================\n")
cat(sprintf("Training AdaBoost  iter = %d | maxdepth = %d | nu = %.2f\n\n",
            best_iter, best_maxdepth, best_nu))

set.seed(42)

ada_final <- caret::train(
  Bankrupt ~ .,
  data      = train,
  method    = "ada",
  trControl = trainControl(
    method     = "none",
    classProbs = TRUE
  ),
  tuneGrid  = data.frame(
    iter     = best_iter,
    maxdepth = best_maxdepth,
    nu       = best_nu
  )
)

cat("Model training complete.\n\n")
print(ada_final)

# =============================================================================
# STEP 4: PREDICTIONS ON TEST SET
# =============================================================================

cat("\n=============================================\n")
cat("STEP 4: PREDICTIONS ON TEST SET\n")
cat("=============================================\n")

# Class predictions
ada_pred_class <- predict(ada_final, newdata = test)

# Probability predictions (required for AUC-ROC)
ada_pred_prob  <- predict(ada_final, newdata = test, type = "prob")[, "Yes"]

cat("Prediction distribution:\n")
print(table(ada_pred_class))

# =============================================================================
# STEP 5: MODEL EVALUATION
# =============================================================================

cat("\n=============================================\n")
cat("STEP 5: MODEL EVALUATION\n")
cat("=============================================\n")

# --- Confusion Matrix --------------------------------------------------------
cm <- confusionMatrix(
  data      = ada_pred_class,
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
  predictor = ada_pred_prob,
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

ADA_COLOUR <- "#1ABC9C"   # teal — distinct from LR (blue), RF (green), NN (purple), SVM (orange)

# --- Confusion Matrix Heatmap -----------------------------------------------
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Predicted", "Actual", "Count")

p_cm <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = Count), size = 8, fontface = "bold", colour = "white") +
  scale_fill_gradient(low = ADA_COLOUR, high = "#0E6655") +
  labs(
    title    = "AdaBoost — Confusion Matrix",
    subtitle = "Test set predictions",
    x = "Predicted Class", y = "Actual Class"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("plot_28_ada_confusion_matrix.png",
       plot = p_cm, width = 6, height = 5, dpi = 150)
cat(">> Saved: plot_28_ada_confusion_matrix.png\n")

# --- ROC Curve --------------------------------------------------------------
roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(colour = ADA_COLOUR, linewidth = 1.2) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  annotate("text", x = 0.65, y = 0.15,
           label = paste0("AUC = ", auc_val),
           size = 5, colour = ADA_COLOUR, fontface = "bold") +
  labs(
    title    = "AdaBoost — ROC Curve",
    subtitle = "Test set performance",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal(base_size = 13)

ggsave("plot_29_ada_roc_curve.png",
       plot = p_roc, width = 6, height = 5, dpi = 150)
cat(">> Saved: plot_29_ada_roc_curve.png\n")

# --- Predicted Probability Distribution -------------------------------------
prob_df <- data.frame(
  Probability = ada_pred_prob,
  Actual      = test$Bankrupt
)

p_prob <- ggplot(prob_df, aes(x = Probability, fill = Actual)) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("No" = "#2ECC71", "Yes" = "#E74C3C"),
                    labels  = c("No" = "Not Bankrupt", "Yes" = "Bankrupt")) +
  labs(
    title    = "AdaBoost — Predicted Probability Distribution",
    subtitle = "Distribution of predicted bankruptcy probabilities by actual class",
    x = "Predicted Probability of Bankruptcy",
    y = "Count", fill = ""
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("plot_30_ada_prob_distribution.png",
       plot = p_prob, width = 8, height = 5, dpi = 150)
cat(">> Saved: plot_30_ada_prob_distribution.png\n")

# --- Hyperparameter Tuning Plot ---------------------------------------------
# Shows how AUC-ROC varies across boosting rounds for each (maxdepth, nu) combo
tune_results <- ada_tuned$results
tune_results$Config <- paste0("depth=", tune_results$maxdepth,
                              " | nu=",  tune_results$nu)

p_tune <- ggplot(tune_results,
                 aes(x = iter, y = ROC,
                     colour = Config, group = Config)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_colour_brewer(palette = "Set2", name = "Config") +
  labs(
    title    = "AdaBoost — Cross-Validation Tuning Results",
    subtitle = "5-fold CV AUC-ROC across boosting rounds and configurations",
    x = "Boosting Iterations (iter)", y = "CV AUC-ROC"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 9))

ggsave("plot_31_ada_tuning.png",
       plot = p_tune, width = 8, height = 5, dpi = 150)
cat(">> Saved: plot_31_ada_tuning.png\n")

# =============================================================================
# STEP 7: SAVE RESULTS FOR MODEL COMPARISON
# =============================================================================

cat("\n=============================================\n")
cat("STEP 7: SAVE RESULTS\n")
cat("=============================================\n")

ada_results <- list(
  model   = ada_final,
  cm      = cm,
  roc     = roc_obj,
  auc     = auc_val,
  metrics = data.frame(
    Model       = "AdaBoost",
    Accuracy    = accuracy,
    Precision   = precision,
    Recall      = recall,
    F1_Score    = f1,
    Specificity = specificity,
    AUC_ROC     = auc_val
  )
)

saveRDS(ada_results, "ada_results.rds")
cat(">> Results saved to: ada_results.rds\n")
cat("   (Can be loaded in model comparison script alongside lr/rf/nn/svm_results.rds)\n")

# =============================================================================
# SUMMARY
# =============================================================================

cat("\n=============================================\n")
cat("ADABOOST — SUMMARY\n")
cat("=============================================\n")
cat(sprintf("Algorithm   : AdaBoost (adaptive boosting)\n"))
cat(sprintf("Base learner: Decision tree (rpart)\n"))
cat(sprintf("Best iter   : %d\n",   best_iter))
cat(sprintf("Best depth  : %d\n",   best_maxdepth))
cat(sprintf("Best nu     : %.2f\n", best_nu))
cat(sprintf("Accuracy    : %.2f%%\n", accuracy))
cat(sprintf("Precision   : %.2f%%\n", precision))
cat(sprintf("Recall      : %.2f%%\n", recall))
cat(sprintf("F1 Score    : %.2f%%\n", f1))
cat(sprintf("Specificity : %.2f%%\n", specificity))
cat(sprintf("AUC-ROC     : %.4f\n",   auc_val))
cat("\nOutput files:\n")
cat("  plot_28_ada_confusion_matrix.png\n")
cat("  plot_29_ada_roc_curve.png\n")
cat("  plot_30_ada_prob_distribution.png\n")
cat("  plot_31_ada_tuning.png\n")
cat("  ada_results.rds\n")
cat("=============================================\n")
