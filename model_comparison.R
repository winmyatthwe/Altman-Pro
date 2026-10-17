# =============================================================================
# MODEL COMPARISON
# Big Data Module - Bankruptcy Prediction Dataset
# =============================================================================
# This script loads results from all three models and produces:
#   1. A side-by-side metrics comparison table
#   2. A combined ROC curve plot
#   3. A metrics bar chart for visual comparison
#   4. A confusion matrix summary
#
# Run AFTER all three model scripts have been run successfully.
# =============================================================================

# --- Install & Load Packages -------------------------------------------------

packages <- c("dplyr", "ggplot2", "tidyr", "pROC", "gridExtra", "knitr")

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

library(dplyr)
library(ggplot2)
library(tidyr)
library(pROC)
library(gridExtra)

# =============================================================================
# STEP 1: LOAD ALL MODEL RESULTS
# =============================================================================

cat("=============================================\n")
cat("MODEL COMPARISON\n")
cat("=============================================\n\n")

lr_results <- readRDS("lr_results.rds")
rf_results <- readRDS("rf_results.rds")
nn_results <- readRDS("nn_results.rds")

cat("All model results loaded successfully.\n\n")

# =============================================================================
# STEP 2: METRICS COMPARISON TABLE
# =============================================================================

cat("=============================================\n")
cat("STEP 2: METRICS COMPARISON TABLE\n")
cat("=============================================\n\n")

comparison_df <- bind_rows(
  lr_results$metrics,
  rf_results$metrics,
  nn_results$metrics
)

# Print formatted table
cat("Full model comparison:\n\n")
print(comparison_df, row.names = FALSE)

# Save as CSV for report
write.csv(comparison_df, "model_comparison.csv", row.names = FALSE)
cat("\n>> Saved: model_comparison.csv\n")

# Identify best model per metric
cat("\nBest model per metric:\n")
metrics_to_check <- c("Accuracy", "Precision", "Recall",
                      "F1_Score", "Specificity", "AUC_ROC")

for (metric in metrics_to_check) {
  best_idx   <- which.max(comparison_df[[metric]])
  best_model <- comparison_df$Model[best_idx]
  best_val   <- comparison_df[[metric]][best_idx]
  cat(sprintf("  %-12s : %s (%.2f)\n", metric, best_model, best_val))
}

# =============================================================================
# STEP 3: COMBINED ROC CURVE
# =============================================================================

cat("\n=============================================\n")
cat("STEP 3: COMBINED ROC CURVE\n")
cat("=============================================\n")

# Build ROC dataframes for each model
label_lr <- paste0("Logistic Regression (AUC = ", lr_results$auc, ")")
label_rf <- paste0("Random Forest (AUC = ", rf_results$auc, ")")
label_nn <- paste0("Neural Network (AUC = ", nn_results$auc, ")")

roc_lr <- data.frame(
  FPR   = 1 - lr_results$roc$specificities,
  TPR   = lr_results$roc$sensitivities,
  Model = label_lr
)

roc_rf <- data.frame(
  FPR   = 1 - rf_results$roc$specificities,
  TPR   = rf_results$roc$sensitivities,
  Model = label_rf
)

roc_nn <- data.frame(
  FPR   = 1 - nn_results$roc$specificities,
  TPR   = nn_results$roc$sensitivities,
  Model = label_nn
)

roc_combined <- bind_rows(roc_lr, roc_rf, roc_nn)

p_roc_combined <- ggplot(roc_combined,
                         aes(x = FPR, y = TPR, colour = Model)) +
  geom_line(linewidth = 1.2) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  scale_colour_manual(values = setNames(
    c("#2980B9", "#27AE60", "#8E44AD"),
    c(label_lr, label_rf, label_nn)
  )) +
  labs(
    title    = "ROC Curve Comparison — All Three Models",
    subtitle = "Higher AUC indicates better overall discriminative ability",
    x        = "False Positive Rate (1 - Specificity)",
    y        = "True Positive Rate (Sensitivity)",
    colour   = ""
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        legend.text     = element_text(size = 9))

ggsave("plot_20_combined_roc.png",
       plot = p_roc_combined, width = 8, height = 6, dpi = 150)
cat(">> Saved: plot_20_combined_roc.png\n")

# =============================================================================
# STEP 4: METRICS BAR CHART
# =============================================================================

cat("\n=============================================\n")
cat("STEP 4: METRICS BAR CHART\n")
cat("=============================================\n")

# Reshape for plotting
metrics_long <- comparison_df %>%
  select(Model, Accuracy, Precision, Recall, F1_Score, Specificity) %>%
  pivot_longer(
    cols      = -Model,
    names_to  = "Metric",
    values_to = "Value"
  ) %>%
  mutate(Metric = factor(Metric,
                         levels = c("Accuracy", "Specificity",
                                    "Precision", "Recall", "F1_Score"),
                         labels = c("Accuracy", "Specificity",
                                    "Precision", "Recall", "F1 Score")))

p_metrics <- ggplot(metrics_long,
                    aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = paste0(round(Value, 1), "%")),
            position = position_dodge(width = 0.7),
            vjust = -0.4, size = 2.8, fontface = "bold") +
  scale_fill_manual(values = c(
    "Logistic Regression" = "#2980B9",
    "Random Forest"       = "#27AE60",
    "Neural Network"      = "#8E44AD"
  )) +
  ylim(0, 115) +
  labs(
    title    = "Model Performance Comparison",
    subtitle = "All metrics calculated on the held-out test set",
    x        = "Metric", y        = "Value (%)", fill = ""
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("plot_21_metrics_comparison.png",
       plot = p_metrics, width = 10, height = 6, dpi = 150)
cat(">> Saved: plot_21_metrics_comparison.png\n")

# =============================================================================
# STEP 5: CONFUSION MATRIX SUMMARY
# =============================================================================

cat("\n=============================================\n")
cat("STEP 5: CONFUSION MATRIX SUMMARY\n")
cat("=============================================\n\n")

extract_cm <- function(results, model_name) {
  cm_table <- results$cm$table
  data.frame(
    Model           = model_name,
    True_Negative   = cm_table["No",  "No"],
    False_Positive  = cm_table["Yes", "No"],
    False_Negative  = cm_table["No",  "Yes"],
    True_Positive   = cm_table["Yes", "Yes"]
  )
}

cm_summary <- bind_rows(
  extract_cm(lr_results, "Logistic Regression"),
  extract_cm(rf_results, "Random Forest"),
  extract_cm(nn_results, "Neural Network")
)

cat("Confusion matrix summary (all models):\n\n")
print(cm_summary, row.names = FALSE)

cat("\nInterpretation:\n")
cat("  True Negative  (TN) : Correctly predicted NOT bankrupt\n")
cat("  False Positive (FP) : Predicted bankrupt, actually NOT bankrupt (false alarm)\n")
cat("  False Negative (FN) : Predicted NOT bankrupt, actually bankrupt (missed!)\n")
cat("  True Positive  (TP) : Correctly predicted bankrupt\n\n")

cat("Bankruptcies caught out of 44 total:\n")
for (i in 1:nrow(cm_summary)) {
  cat(sprintf("  %-22s : %d / 44 (%.1f%%)\n",
              cm_summary$Model[i],
              cm_summary$True_Positive[i],
              cm_summary$True_Positive[i] / 44 * 100))
}

cat("\nFalse alarms raised:\n")
for (i in 1:nrow(cm_summary)) {
  cat(sprintf("  %-22s : %d false alarms\n",
              cm_summary$Model[i],
              cm_summary$False_Positive[i]))
}

write.csv(cm_summary, "confusion_matrix_summary.csv", row.names = FALSE)
cat("\n>> Saved: confusion_matrix_summary.csv\n")

# =============================================================================
# STEP 6: FINAL RECOMMENDATION
# =============================================================================

cat("\n=============================================\n")
cat("STEP 6: FINAL RECOMMENDATION\n")
cat("=============================================\n\n")

cat("Based on the evaluation metrics:\n\n")

cat("  BEST OVERALL MODEL     : Random Forest\n")
cat("  Reason                 : Highest accuracy (93.69%), F1 Score (41.89%)\n")
cat("                           and Specificity (94.47%). Fewest false alarms.\n\n")

cat("  BEST FOR RISK AVERSION : Logistic Regression\n")
cat("  Reason                 : Highest Recall (90.91%) and AUC-ROC (0.9414).\n")
cat("                           Catches most actual bankruptcies (40/44).\n\n")

cat("  NEURAL NETWORK         : Competitive but underperforms both models.\n")
cat("  Reason                 : High training error suggests incomplete\n")
cat("                           convergence. Would benefit from a larger\n")
cat("                           dataset or more advanced optimiser (e.g. Adam).\n\n")

cat("NOTE FOR REPORT: The choice of best model depends on the business\n")
cat("objective. In credit risk, missing a real bankruptcy (false negative)\n")
cat("is more costly than a false alarm (false positive). Under this\n")
cat("criterion, Logistic Regression's superior recall is most valuable.\n")

# =============================================================================
# SUMMARY
# =============================================================================

cat("\n=============================================\n")
cat("MODEL COMPARISON COMPLETE\n")
cat("=============================================\n")
cat("Output files:\n")
cat("  model_comparison.csv\n")
cat("  confusion_matrix_summary.csv\n")
cat("  plot_20_combined_roc.png\n")
cat("  plot_21_metrics_comparison.png\n")
cat("=============================================\n")