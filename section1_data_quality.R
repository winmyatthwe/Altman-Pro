# =============================================================================
# SECTION 1: DATA QUALITY CONTROL
# Big Data Module - Bankruptcy Prediction Dataset
# =============================================================================

# --- 1.0 Install & Load Required Packages ------------------------------------

packages <- c("readxl", "dplyr", "ggplot2", "tidyr", "corrplot", "scales", "gridExtra")

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

library(readxl)
library(dplyr)
library(ggplot2)
library(tidyr)
library(corrplot)
library(scales)
library(gridExtra)

# =============================================================================
# 1.1 LOAD DATA & INITIAL INSPECTION
# =============================================================================

df <- read_excel("Group 2 dataset.xlsx")

# Clean column names: trim whitespace, replace special characters with dots
colnames(df) <- colnames(df) %>%
  trimws() %>%
  make.names()

# Rename target variable
colnames(df)[1] <- "Bankrupt"
df$Bankrupt <- as.factor(df$Bankrupt)

cat("=============================================\n")
cat("1.1 DATASET SIZE & STRUCTURE\n")
cat("=============================================\n")
cat("Rows (companies)   :", nrow(df), "\n")
cat("Columns (features) :", ncol(df), "\n\n")

# Count feature types
num_features    <- sum(sapply(df, is.numeric))
factor_features <- sum(sapply(df, is.factor))
cat("Numeric features :", num_features, "\n")
cat("Factor features  :", factor_features, "\n\n")

cat("Column names (first 10):\n")
print(colnames(df)[1:10])

# =============================================================================
# 1.2 MISSING VALUES
# =============================================================================

cat("\n=============================================\n")
cat("1.2 MISSING VALUES\n")
cat("=============================================\n")

missing_counts <- colSums(is.na(df))
total_missing  <- sum(missing_counts)

cat("Total missing values:", total_missing, "\n")
cat("Columns with any missing values:", sum(missing_counts > 0), "\n")

if (total_missing == 0) {
  cat(">> Dataset is complete - no imputation required.\n")
} else {
  cat("\nColumns with missing values:\n")
  print(sort(missing_counts[missing_counts > 0], decreasing = TRUE))
}

# =============================================================================
# 1.3 CLASS DISTRIBUTION (TARGET VARIABLE)
# =============================================================================

cat("\n=============================================\n")
cat("1.3 TARGET VARIABLE - CLASS DISTRIBUTION\n")
cat("=============================================\n")

class_counts <- table(df$Bankrupt)
class_pct    <- round(prop.table(class_counts) * 100, 2)

cat("Class counts:\n")
print(class_counts)
cat("\nClass percentages:\n")
print(class_pct)

# Plot
class_df <- data.frame(
  Class = c("Not Bankrupt (0)", "Bankrupt (1)"),
  Count = as.numeric(class_counts),
  Pct   = as.numeric(class_pct)
)

p_class <- ggplot(class_df, aes(x = Class, y = Count, fill = Class)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_text(aes(label = paste0(Count, "\n(", Pct, "%)")),
            vjust = -0.3, size = 5, fontface = "bold") +
  scale_fill_manual(values = c("Not Bankrupt (0)" = "#2ECC71",
                               "Bankrupt (1)"     = "#E74C3C")) +
  ylim(0, max(class_df$Count) * 1.15) +
  labs(title    = "Class Distribution: Bankrupt vs Non-Bankrupt",
       subtitle = "Severe class imbalance: 96.8% non-bankrupt vs 3.2% bankrupt",
       x = "", y = "Number of Companies") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("plot_01_class_distribution.png", plot = p_class, width = 7, height = 5, dpi = 150)
cat(">> Saved: plot_01_class_distribution.png\n")

# =============================================================================
# 1.4 SUMMARY STATISTICS
# =============================================================================

cat("\n=============================================\n")
cat("1.4 SUMMARY STATISTICS\n")
cat("=============================================\n")

numeric_df <- df %>% select(where(is.numeric))

stats_df <- data.frame(
  Feature = names(numeric_df),
  Mean    = round(sapply(numeric_df, mean,   na.rm = TRUE), 4),
  Median  = round(sapply(numeric_df, median, na.rm = TRUE), 4),
  SD      = round(sapply(numeric_df, sd,     na.rm = TRUE), 4),
  Min     = round(sapply(numeric_df, min,    na.rm = TRUE), 4),
  Max     = round(sapply(numeric_df, max,    na.rm = TRUE), 4),
  row.names = NULL
)

cat("Summary statistics (first 10 features shown):\n")
print(head(stats_df, 10))

write.csv(stats_df, "summary_statistics.csv", row.names = FALSE)
cat("\n>> Full summary saved to: summary_statistics.csv\n")

# =============================================================================
# 1.5 OUTLIER DETECTION (IQR Method)
# =============================================================================

cat("\n=============================================\n")
cat("1.5 OUTLIER DETECTION\n")
cat("=============================================\n")

count_outliers <- function(x) {
  Q1      <- quantile(x, 0.25, na.rm = TRUE)
  Q3      <- quantile(x, 0.75, na.rm = TRUE)
  IQR_val <- Q3 - Q1
  sum(x < (Q1 - 1.5 * IQR_val) | x > (Q3 + 1.5 * IQR_val), na.rm = TRUE)
}

outlier_counts <- sapply(numeric_df, count_outliers)
outlier_pct    <- round(outlier_counts / nrow(df) * 100, 2)

outlier_df <- data.frame(
  Feature       = names(outlier_counts),
  Outlier_Count = as.numeric(outlier_counts),
  Outlier_Pct   = as.numeric(outlier_pct),
  row.names     = NULL
) %>% arrange(desc(Outlier_Count))

cat("Top 15 features with most outliers:\n")
print(head(outlier_df, 15))

# Bar chart - top 15
top15 <- head(outlier_df, 15)

p_outliers <- ggplot(top15, aes(x = reorder(Feature, Outlier_Pct), y = Outlier_Pct)) +
  geom_bar(stat = "identity", fill = "#F39C12") +
  geom_text(aes(label = paste0(Outlier_Pct, "%")), hjust = -0.1, size = 3) +
  coord_flip() +
  ylim(0, max(top15$Outlier_Pct) * 1.15) +
  labs(title    = "Top 15 Features by Outlier Percentage",
       subtitle = "IQR method: values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR",
       x = "Feature", y = "Outliers (% of all rows)") +
  theme_minimal(base_size = 10)

ggsave("plot_02_outliers.png", plot = p_outliers, width = 11, height = 7, dpi = 150)
cat(">> Saved: plot_02_outliers.png\n")

# Boxplots for top 6 outlier features grouped by Bankrupt
top6_cols <- as.character(head(outlier_df$Feature, 6))

df_box <- df %>%
  select(Bankrupt, all_of(top6_cols)) %>%
  pivot_longer(cols = -Bankrupt, names_to = "Feature", values_to = "Value")

p_box <- ggplot(df_box, aes(x = Bankrupt, y = Value, fill = Bankrupt)) +
  geom_boxplot(outlier.size = 0.4, outlier.alpha = 0.3) +
  facet_wrap(~Feature, scales = "free_y", ncol = 3) +
  scale_fill_manual(values = c("0" = "#2ECC71", "1" = "#E74C3C"),
                    labels  = c("0" = "Not Bankrupt", "1" = "Bankrupt")) +
  labs(title    = "Boxplots: Top 6 Features with Most Outliers",
       subtitle = "Grouped by bankruptcy status",
       x = "Bankrupt", y = "Value", fill = "") +
  theme_minimal(base_size = 10) +
  theme(legend.position = "bottom")

ggsave("plot_03_boxplots.png", plot = p_box, width = 13, height = 9, dpi = 150)
cat(">> Saved: plot_03_boxplots.png\n")

# =============================================================================
# 1.6 FEATURE DISTRIBUTIONS (Key Financial Ratios)
# =============================================================================

cat("\n=============================================\n")
cat("1.6 FEATURE DISTRIBUTIONS\n")
cat("=============================================\n")

# Exact names after make.names() cleaning
key_features <- c(
  "ROA.C.before.interest.and.depreciation.before.interest",
  "Operating.Gross.Margin",
  "Debt.ratio",
  "Current.Ratio",
  "Net.worth.Assets",
  "Cash.Flow.to.Sales",
  "Total.Asset.Turnover",
  "Net.Income.to.Total.Assets",
  "Working.Capital.to.Total.Assets"
)

# Keep only columns that exist in the dataframe
key_features <- key_features[key_features %in% names(df)]
cat("Plotting distributions for", length(key_features), "key features.\n")

df_hist <- df %>%
  select(Bankrupt, all_of(key_features)) %>%
  pivot_longer(cols = -Bankrupt, names_to = "Feature", values_to = "Value")

p_hist <- ggplot(df_hist, aes(x = Value, fill = Bankrupt)) +
  geom_histogram(bins = 60, alpha = 0.6, position = "identity") +
  facet_wrap(~Feature, scales = "free", ncol = 3) +
  scale_fill_manual(values = c("0" = "#3498DB", "1" = "#E74C3C"),
                    labels  = c("0" = "Not Bankrupt", "1" = "Bankrupt")) +
  labs(title    = "Distributions of Key Financial Ratios",
       subtitle = "Bankrupt vs Non-Bankrupt companies",
       x = "Value", y = "Count", fill = "") +
  theme_minimal(base_size = 9) +
  theme(legend.position = "bottom")

ggsave("plot_04_distributions.png", plot = p_hist, width = 14, height = 11, dpi = 150)
cat(">> Saved: plot_04_distributions.png\n")

# =============================================================================
# 1.7 CORRELATION HEATMAP (Top 20 features most correlated with Bankrupt)
# =============================================================================

cat("\n=============================================\n")
cat("1.7 CORRELATION ANALYSIS\n")
cat("=============================================\n")

df_num <- df %>%
  mutate(Bankrupt_num = as.numeric(as.character(Bankrupt))) %>%
  select(where(is.numeric))

# Correlation of every feature with the target
cor_target <- cor(df_num, df_num$Bankrupt_num, use = "complete.obs")
cor_target  <- cor_target[rownames(cor_target) != "Bankrupt_num", , drop = FALSE]

# Top 20 by absolute correlation
top20_names <- names(sort(abs(cor_target[, 1]), decreasing = TRUE)[1:20])

cat("Top 10 features most correlated with Bankrupt:\n")
print(round(sort(cor_target[top20_names, 1], decreasing = TRUE), 4))

# Correlation matrix among those 20
cor_mat <- cor(df_num[, top20_names], use = "complete.obs")

png("plot_05_correlation_matrix.png", width = 1600, height = 1400, res = 130)
corrplot(cor_mat,
         method      = "color",
         type        = "upper",
         tl.cex      = 0.7,
         tl.col      = "black",
         addCoef.col = "black",
         number.cex  = 0.45,
         col         = colorRampPalette(c("#E74C3C", "white", "#2ECC71"))(200),
         title       = "Correlation Matrix: Top 20 Features (by correlation with Bankrupt)",
         mar         = c(0, 0, 2, 0))
dev.off()
cat(">> Saved: plot_05_correlation_matrix.png\n")

# =============================================================================
# COMPLETE
# =============================================================================

cat("\n=============================================\n")
cat("SECTION 1 COMPLETE — OUTPUT FILES:\n")
cat("=============================================\n")
cat("  plot_01_class_distribution.png\n")
cat("  plot_02_outliers.png\n")
cat("  plot_03_boxplots.png\n")
cat("  plot_04_distributions.png\n")
cat("  plot_05_correlation_matrix.png\n")
cat("  summary_statistics.csv\n")
cat("=============================================\n")

