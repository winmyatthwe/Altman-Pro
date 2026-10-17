# =============================================================================
# FEATURE EXTRACTION — All Three Models
# Big Data Module - Bankruptcy Prediction Dataset
# =============================================================================
# This script extracts and displays all features used across all three models,
# including their original names, cleaned names, and which selection method
# included them (Random Forest importance or forced binary flag).
# =============================================================================

# --- Install & Load Packages -------------------------------------------------

packages <- c("dplyr", "ggplot2", "randomForest")

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

library(dplyr)
library(ggplot2)
library(randomForest)

# =============================================================================
# STEP 1: LOAD DATA & RESULTS
# =============================================================================

cat("=============================================\n")
cat("FEATURE EXTRACTION — ALL MODELS\n")
cat("=============================================\n\n")

# Load final dataset (contains only selected features)
df_final  <- readRDS("df_final.rds")
train     <- readRDS("train_scaled.rds")
lr_results <- readRDS("lr_results.rds")
rf_results <- readRDS("rf_results.rds")
nn_results <- readRDS("nn_results.rds")

# =============================================================================
# STEP 2: EXTRACT FEATURE NAMES
# =============================================================================

cat("=============================================\n")
cat("STEP 2: FEATURES USED IN ALL MODELS\n")
cat("=============================================\n\n")

# All three models use the same feature set from df_final
all_features <- names(df_final %>% select(-Bankrupt))

# Identify which were forced in on theoretical grounds
binary_flags <- c("Liability.Assets.Flag", "Net.Income.Flag")
forced_in    <- binary_flags[binary_flags %in% all_features]
rf_selected  <- all_features[!all_features %in% forced_in]

cat(sprintf("Total features used across all models: %d\n\n", length(all_features)))
cat(sprintf("  Selected by Random Forest importance : %d\n", length(rf_selected)))
cat(sprintf("  Forced in on theoretical grounds     : %d\n\n", length(forced_in)))

# =============================================================================
# STEP 3: FULL FEATURE TABLE WITH IMPORTANCE SCORES
# =============================================================================

cat("=============================================\n")
cat("STEP 3: FULL FEATURE TABLE\n")
cat("=============================================\n\n")

# Get RF importance scores from final model
rf_model     <- rf_results$model
importance_scores <- data.frame(
  Feature          = rownames(importance(rf_model)),
  RF_Gini_Importance = round(importance(rf_model)[, "MeanDecreaseGini"], 4),
  row.names        = NULL
)

# Get LR coefficients
lr_model   <- lr_results$model$finalModel
lr_coefs   <- data.frame(
  Feature = names(coef(lr_model))[-1],  # exclude intercept
  LR_Coefficient = round(coef(lr_model)[-1], 4),
  row.names = NULL
)

# Build master feature table
feature_table <- data.frame(
  Rank          = seq_along(all_features),
  Feature       = all_features,
  Selection     = ifelse(all_features %in% forced_in,
                         "Forced (theoretical)",
                         "Random Forest Importance"),
  stringsAsFactors = FALSE
)

# Merge RF importance scores
feature_table <- feature_table %>%
  left_join(importance_scores, by = "Feature")

# Merge LR coefficients
feature_table <- feature_table %>%
  left_join(lr_coefs, by = "Feature")

# Sort by RF importance descending
feature_table <- feature_table %>%
  arrange(desc(RF_Gini_Importance))

feature_table$Rank <- seq_len(nrow(feature_table))

cat("Feature table (sorted by Random Forest importance):\n\n")
print(feature_table, row.names = FALSE)

# Save to CSV
write.csv(feature_table, "feature_table.csv", row.names = FALSE)
cat("\n>> Saved: feature_table.csv\n")

# =============================================================================
# STEP 4: FEATURE IMPORTANCE PLOT (Final RF Model)
# =============================================================================

cat("\n=============================================\n")
cat("STEP 4: FEATURE IMPORTANCE PLOT\n")
cat("=============================================\n")

plot_df <- feature_table %>%
  arrange(RF_Gini_Importance) %>%
  mutate(
    Feature = factor(Feature, levels = Feature),
    Selection = factor(Selection)
  )

p_features <- ggplot(plot_df,
                     aes(x = Feature,
                         y = RF_Gini_Importance,
                         fill = Selection)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(RF_Gini_Importance, 1)),
            hjust = -0.1, size = 3) +
  coord_flip() +
  ylim(0, max(plot_df$RF_Gini_Importance, na.rm = TRUE) * 1.15) +
  scale_fill_manual(values = c(
    "Random Forest Importance" = "#2980B9",
    "Forced (theoretical)"    = "#E74C3C"
  )) +
  labs(
    title    = "Features Used Across All Three Models",
    subtitle = "Blue = selected by RF importance | Red = forced in on theoretical grounds",
    x        = "Feature",
    y        = "Random Forest Gini Importance",
    fill     = "Selection Method"
  ) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "bottom")

ggsave("plot_22_all_features.png",
       plot = p_features, width = 12, height = 9, dpi = 150)
cat(">> Saved: plot_22_all_features.png\n")

# =============================================================================
# STEP 5: LOGISTIC REGRESSION COEFFICIENTS PLOT
# =============================================================================

cat("\n=============================================\n")
cat("STEP 5: LOGISTIC REGRESSION COEFFICIENTS\n")
cat("=============================================\n\n")

# LR coefficients show direction of effect (positive = increases bankruptcy risk)
lr_coef_df <- feature_table %>%
  filter(!is.na(LR_Coefficient)) %>%
  arrange(LR_Coefficient) %>%
  mutate(
    Feature   = factor(Feature, levels = Feature),
    Direction = ifelse(LR_Coefficient > 0, "Increases Risk", "Decreases Risk")
  )

cat("Logistic Regression coefficients:\n")
print(lr_coef_df %>% select(Feature, LR_Coefficient, Direction),
      row.names = FALSE)

p_coef <- ggplot(lr_coef_df,
                 aes(x = Feature, y = LR_Coefficient, fill = Direction)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("Increases Risk" = "#E74C3C",
                               "Decreases Risk" = "#2ECC71")) +
  labs(
    title    = "Logistic Regression Coefficients",
    subtitle = "Positive = increases bankruptcy probability | Negative = decreases it",
    x        = "Feature",
    y        = "Coefficient",
    fill     = ""
  ) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "bottom")

ggsave("plot_23_lr_coefficients.png",
       plot = p_coef, width = 12, height = 9, dpi = 150)
cat(">> Saved: plot_23_lr_coefficients.png\n")

# =============================================================================
# STEP 6: PLAIN ENGLISH FEATURE SUMMARY
# =============================================================================

cat("\n=============================================\n")
cat("STEP 6: PLAIN ENGLISH FEATURE SUMMARY\n")
cat("=============================================\n\n")

cat("FEATURES SELECTED BY RANDOM FOREST IMPORTANCE:\n")
cat("-----------------------------------------------\n")
for (i in seq_along(rf_selected)) {
  cat(sprintf("  %2d. %s\n", i, rf_selected[i]))
}

cat("\nFEATURES FORCED IN ON THEORETICAL GROUNDS:\n")
cat("-------------------------------------------\n")
for (f in forced_in) {
  if (f == "Liability.Assets.Flag") {
    cat(sprintf("  - %s\n", f))
    cat("    (1 if Total Liabilities > Total Assets — signals insolvency)\n")
    cat("    (Altman, 1968; Strobl et al., 2007)\n\n")
  }
  if (f == "Net.Income.Flag") {
    cat(sprintf("  - %s\n", f))
    cat("    (1 if Net Income negative for 2 consecutive years)\n")
    cat("    (Zmijewski, 1984; Strobl et al., 2007)\n\n")
  }
}

cat("ALL THREE MODELS USE THESE SAME", length(all_features), "FEATURES.\n")
cat("Features are identical across Logistic Regression,\n")
cat("Random Forest, and Neural Network — ensuring a fair comparison.\n")

# =============================================================================
# SUMMARY
# =============================================================================

cat("\n=============================================\n")
cat("FEATURE EXTRACTION COMPLETE\n")
cat("=============================================\n")
cat("Total features         :", length(all_features), "\n")
cat("RF-selected features   :", length(rf_selected), "\n")
cat("Forced binary flags    :", length(forced_in), "\n")
cat("\nOutput files:\n")
cat("  feature_table.csv\n")
cat("  plot_22_all_features.png\n")
cat("  plot_23_lr_coefficients.png\n")
cat("=============================================\n")