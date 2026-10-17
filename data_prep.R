# =============================================================================
# DATA PREPARATION - Section 2 Setup
# Big Data Module - Bankruptcy Prediction Dataset
# =============================================================================
# This script must be run AFTER feature_selection.R
# It prepares the final dataset for model training by:
#   1. Loading the selected features (df_final.rds)
#   2. Splitting into train/test sets (80/20)
#   3. Applying SMOTE to the training set only
#   4. Scaling/normalising all numeric features
# =============================================================================

# --- Install & Load Packages -------------------------------------------------

packages <- c("dplyr", "caret", "smotefamily", "ggplot2")

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

library(dplyr)
library(caret)
library(smotefamily)
library(ggplot2)

# =============================================================================
# STEP 1: LOAD FEATURE-SELECTED DATASET
# =============================================================================

cat("=============================================\n")
cat("STEP 1: LOAD DATA\n")
cat("=============================================\n")

# Load the dataset saved by feature_selection.R
df_final <- readRDS("df_final.rds")

cat("Loaded df_final:", nrow(df_final), "rows x", ncol(df_final), "columns\n")
cat("Features:", paste(names(df_final)[-1], collapse = ", "), "\n\n")

cat("Class distribution before preparation:\n")
print(table(df_final$Bankrupt))
print(round(prop.table(table(df_final$Bankrupt)) * 100, 2))

# =============================================================================
# STEP 2: TRAIN / TEST SPLIT (80/20)
# =============================================================================
# We split BEFORE any balancing or scaling.
# This ensures the test set reflects real-world class distribution
# and no information from the test set leaks into training.

cat("\n=============================================\n")
cat("STEP 2: TRAIN / TEST SPLIT (80/20)\n")
cat("=============================================\n")

set.seed(42)  # for reproducibility

train_index <- createDataPartition(
  df_final$Bankrupt,
  p    = 0.80,
  list = FALSE
)

train_raw <- df_final[ train_index, ]
test_raw  <- df_final[-train_index, ]

cat("Training set:", nrow(train_raw), "rows\n")
cat("Test set    :", nrow(test_raw), "rows\n\n")

cat("Training set class distribution:\n")
print(table(train_raw$Bankrupt))

cat("\nTest set class distribution:\n")
print(table(test_raw$Bankrupt))

# Note: createDataPartition does stratified sampling, so class proportions
# are preserved in both train and test sets

# =============================================================================
# STEP 3: SMOTE - Handle Class Imbalance (Training Set Only)
# =============================================================================
# SMOTE (Synthetic Minority Oversampling Technique) generates synthetic
# examples of the minority class (bankrupt = 1) by interpolating between
# existing minority class samples and their nearest neighbours.
#
# CRITICAL: SMOTE is applied ONLY to the training set.
# Applying it to the test set would give artificially inflated performance
# metrics that don't reflect real-world conditions.

cat("\n=============================================\n")
cat("STEP 3: SMOTE (Training Set Only)\n")
cat("=============================================\n")

# SMOTE requires numeric features and a separate target vector
# Separate features and target
train_features <- train_raw %>% select(-Bankrupt)
train_target   <- as.numeric(as.character(train_raw$Bankrupt))

# Apply SMOTE
# K = 5 nearest neighbours (default, standard setting)
# dup_size = 0 means auto-balance to roughly 50/50
set.seed(42)
smote_result <- SMOTE(
  X        = train_features,
  target   = train_target,
  K        = 5,
  dup_size = 0
)

# Rebuild training dataframe
train_smote <- smote_result$data
colnames(train_smote)[ncol(train_smote)] <- "Bankrupt"
train_smote$Bankrupt <- as.factor(train_smote$Bankrupt)

cat("Before SMOTE - Training set:\n")
print(table(train_raw$Bankrupt))

cat("\nAfter SMOTE - Training set:\n")
print(table(train_smote$Bankrupt))
print(round(prop.table(table(train_smote$Bankrupt)) * 100, 2))

# Visualise class balance before and after SMOTE
balance_df <- data.frame(
  Stage = rep(c("Before SMOTE", "After SMOTE"), each = 2),
  Class = rep(c("Not Bankrupt (0)", "Bankrupt (1)"), 2),
  Count = c(
    as.numeric(table(train_raw$Bankrupt)),
    as.numeric(table(train_smote$Bankrupt))
  )
)
balance_df$Stage <- factor(balance_df$Stage,
                           levels = c("Before SMOTE", "After SMOTE"))

p_smote <- ggplot(balance_df, aes(x = Class, y = Count, fill = Class)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_text(aes(label = Count), vjust = -0.4, size = 4, fontface = "bold") +
  facet_wrap(~Stage) +
  scale_fill_manual(values = c("Not Bankrupt (0)" = "#2ECC71",
                               "Bankrupt (1)"     = "#E74C3C")) +
  ylim(0, max(balance_df$Count) * 1.15) +
  labs(
    title    = "Class Balance: Before and After SMOTE",
    subtitle = "SMOTE applied to training set only",
    x = "", y = "Number of Companies"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

ggsave("plot_08_smote_balance.png",
       plot = p_smote, width = 9, height = 5, dpi = 150)
cat("\n>> Saved: plot_08_smote_balance.png\n")

# =============================================================================
# STEP 4: FEATURE SCALING (Standardisation)
# =============================================================================
# Standardisation transforms each feature to have mean = 0 and SD = 1.
# This is required for Logistic Regression and Neural Networks which are
# sensitive to the scale of input features.
# Random Forest does not require scaling but we apply it consistently
# so all three models use the same prepared dataset.
#
# CRITICAL: The scaler is FIT on the training set only, then APPLIED
# to both train and test sets. This prevents data leakage — we cannot
# let test set statistics influence the scaling parameters.

cat("\n=============================================\n")
cat("STEP 4: FEATURE SCALING (Standardisation)\n")
cat("=============================================\n")

# Identify numeric feature columns (exclude target and binary flags)
# Binary flags (0/1) should NOT be scaled — they would lose their meaning
binary_cols  <- c("Liability.Assets.Flag", "Net.Income.Flag")
binary_cols  <- binary_cols[binary_cols %in% names(train_smote)]

numeric_cols <- train_smote %>%
  select(-Bankrupt, -all_of(binary_cols)) %>%
  select(where(is.numeric)) %>%
  names()

cat("Scaling", length(numeric_cols), "numeric features\n")
cat("Leaving unscaled (binary flags):", paste(binary_cols, collapse = ", "), "\n\n")

# Fit scaler on TRAINING set only
scaler <- preProcess(
  train_smote[, numeric_cols],
  method = c("center", "scale")
)

# Apply scaler to training set
train_scaled <- train_smote
train_scaled[, numeric_cols] <- predict(scaler, train_smote[, numeric_cols])

# Apply same scaler to test set (using training set parameters)
test_scaled <- test_raw
test_scaled[, numeric_cols] <- predict(scaler, test_raw[, numeric_cols])

cat("Scaling complete.\n")
cat("Training set (after SMOTE + scaling):", nrow(train_scaled), "rows\n")
cat("Test set (original distribution + scaling):", nrow(test_scaled), "rows\n\n")

# Verify scaling worked - means should be ~0, SDs should be ~1
cat("Verification - first 5 features (train):\n")
verify <- train_scaled %>%
  select(all_of(numeric_cols[1:5])) %>%
  summarise(across(everything(), list(
    mean = ~round(mean(.), 3),
    sd   = ~round(sd(.), 3)
  )))
print(verify)

# =============================================================================
# STEP 5: SAVE PREPARED DATASETS
# =============================================================================

cat("\n=============================================\n")
cat("STEP 5: SAVE PREPARED DATASETS\n")
cat("=============================================\n")

# Save all prepared objects for use in model scripts
saveRDS(train_scaled, "train_scaled.rds")   # SMOTE + scaled training set
saveRDS(test_scaled,  "test_scaled.rds")    # Scaled test set (original distribution)
saveRDS(scaler,       "scaler.rds")         # Scaler object (for future predictions)

cat("Saved:\n")
cat("  train_scaled.rds  - SMOTE-balanced, scaled training set\n")
cat("  test_scaled.rds   - Scaled test set (real-world class distribution)\n")
cat("  scaler.rds        - Scaler parameters (mean & SD from training set)\n")

# =============================================================================
# SUMMARY
# =============================================================================

cat("\n=============================================\n")
cat("DATA PREPARATION COMPLETE\n")
cat("=============================================\n")
cat("Original dataset      :", nrow(df_final), "rows\n")
cat("Training set (raw)    :", nrow(train_raw), "rows\n")
cat("Training set (SMOTE)  :", nrow(train_smote), "rows\n")
cat("Test set              :", nrow(test_raw), "rows\n")
cat("Features used         :", ncol(train_scaled) - 1, "\n")
cat("Scaled features       :", length(numeric_cols), "\n")
cat("Unscaled (binary)     :", length(binary_cols), "\n")
cat("\nReady for model training.\n")
cat("Load in model scripts with:\n")
cat("  train <- readRDS('train_scaled.rds')\n")
cat("  test  <- readRDS('test_scaled.rds')\n")
cat("=============================================\n")