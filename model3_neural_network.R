# =============================================================================
# MODEL 3: NEURAL NETWORK
# Big Data Module - Bankruptcy Prediction Dataset
# =============================================================================
# A feedforward neural network is a deep learning technique that learns
# complex non-linear relationships between features through multiple layers
# of interconnected nodes. Neural networks have shown strong performance
# in financial distress prediction (Tam & Kiang, 1992; Zhang et al., 1999).
#
# Architecture: Input Layer -> Hidden Layer(s) -> Output Layer
#
# Run AFTER: feature_selection.R and data_preparation.R
# =============================================================================

# --- Install & Load Packages -------------------------------------------------

packages <- c("dplyr", "neuralnet", "caret", "pROC", "ggplot2")

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

library(dplyr)
library(neuralnet)
library(caret)
library(pROC)
library(ggplot2)

# =============================================================================
# STEP 1: LOAD PREPARED DATA
# =============================================================================

cat("=============================================\n")
cat("MODEL 3: NEURAL NETWORK\n")
cat("=============================================\n\n")

train <- readRDS("train_scaled.rds")
test  <- readRDS("test_scaled.rds")

# neuralnet requires a numeric binary target (0/1), not a factor
# as.character() handles both "1"/"0" and "Yes"/"No" factor encodings
train_nn <- train %>%
  mutate(Bankrupt = ifelse(as.character(Bankrupt) %in% c("1", "Yes"), 1, 0))
test_nn <- test %>%
  mutate(Bankrupt = ifelse(as.character(Bankrupt) %in% c("1", "Yes"), 1, 0))

cat("Training set:", nrow(train_nn), "rows |",
    "Bankrupt (1):", sum(train_nn$Bankrupt),
    "| Not Bankrupt (0):", sum(train_nn$Bankrupt == 0), "\n")
cat("Test set    :", nrow(test_nn),  "rows |",
    "Bankrupt (1):", sum(test_nn$Bankrupt),
    "| Not Bankrupt (0):", sum(test_nn$Bankrupt == 0), "\n\n")

# =============================================================================
# STEP 2: BUILD MODEL FORMULA
# =============================================================================
# neuralnet requires an explicit formula listing all features
# We build this programmatically from column names
 
cat("=============================================\n")
cat("STEP 2: PREPARING MODEL FORMULA\n")
cat("=============================================\n")
 
feature_names <- names(train_nn %>% select(-Bankrupt))
nn_formula    <- as.formula(
  paste("Bankrupt ~", paste(feature_names, collapse = " + "))
)
 
cat("Features used:", length(feature_names), "\n")
cat("Formula built successfully.\n\n")
 
# =============================================================================
# STEP 3: TRAIN NEURAL NETWORK
# =============================================================================
# Architecture: 2 hidden layers with 16 and 8 nodes respectively
#   - Input layer  : one node per feature
#   - Hidden layer 1: 16 nodes with sigmoid activation
#   - Hidden layer 2: 8 nodes with sigmoid activation
#   - Output layer : 1 node (probability of bankruptcy)
#
# This architecture progressively compresses the information,
# forcing the network to learn the most important representations.
#
# Hyperparameters:
#   hidden      = c(16, 8)  — two hidden layers
#   threshold   = 0.01      — convergence criterion (lower = more precise)
#   stepmax     = 1e6       — max training iterations
#   learningrate= 0.01      — how fast weights are updated
#   act.fct     = "logistic"— sigmoid activation function
 
cat("=============================================\n")
cat("STEP 3: TRAINING NEURAL NETWORK\n")
cat("=============================================\n")
cat("Architecture: Input ->  16 nodes -> 8 nodes -> Output\n")
cat("This may take several minutes...\n\n")
 
set.seed(42)
 
nn_model <- neuralnet(
  formula      = nn_formula,
  data         = train_nn,
  hidden       = c(16, 8),
  linear.output = FALSE,     # FALSE for classification (sigmoid output)
  act.fct      = "logistic",
  threshold    = 0.01,
  stepmax      = 1e6,
  learningrate = 0.01
)
 
cat("Model training complete.\n\n")
cat("Training steps taken:", nn_model$result.matrix["steps", 1], "\n")
cat("Final error         :", round(nn_model$result.matrix["error", 1], 6), "\n\n")
 
# =============================================================================
# STEP 4: PREDICTIONS ON TEST SET
# =============================================================================
 
cat("=============================================\n")
cat("STEP 4: PREDICTIONS ON TEST SET\n")
cat("=============================================\n")
 
# Generate predicted probabilities
nn_pred_prob <- predict(nn_model,
                        newdata = test_nn %>% select(-Bankrupt))[, 1]
 
# Convert probabilities to class predictions using 0.5 threshold
nn_pred_class <- ifelse(nn_pred_prob >= 0.5, 1, 0)
 
cat("Prediction distribution:\n")
print(table(nn_pred_class))
cat("\n")
 
# =============================================================================
# STEP 5: MODEL EVALUATION
# =============================================================================
 
cat("=============================================\n")
cat("STEP 5: MODEL EVALUATION\n")
cat("=============================================\n")
 
# Convert to factors for confusionMatrix
nn_pred_factor  <- factor(nn_pred_class,  levels = c(0, 1), labels = c("No", "Yes"))
test_nn_factor  <- factor(test_nn$Bankrupt, levels = c(0, 1), labels = c("No", "Yes"))
 
# --- Confusion Matrix --------------------------------------------------------
cm <- confusionMatrix(
  data      = nn_pred_factor,
  reference = test_nn_factor,
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
  response  = test_nn$Bankrupt,
  predictor = nn_pred_prob
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
  scale_fill_gradient(low = "#8E44AD", high = "#4A235A") +
  labs(
    title    = "Neural Network — Confusion Matrix",
    subtitle = "Test set predictions",
    x = "Predicted Class", y = "Actual Class"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")
 
ggsave("plot_16_nn_confusion_matrix.png",
       plot = p_cm, width = 6, height = 5, dpi = 150)
cat(">> Saved: plot_16_nn_confusion_matrix.png\n")
 
# --- ROC Curve --------------------------------------------------------------
roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)
 
p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(colour = "#8E44AD", linewidth = 1.2) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  annotate("text", x = 0.65, y = 0.15,
           label = paste0("AUC = ", auc_val),
           size = 5, colour = "#8E44AD", fontface = "bold") +
  labs(
    title    = "Neural Network — ROC Curve",
    subtitle = "Test set performance",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal(base_size = 13)
 
ggsave("plot_17_nn_roc_curve.png",
       plot = p_roc, width = 6, height = 5, dpi = 150)
cat(">> Saved: plot_17_nn_roc_curve.png\n")
 
# --- Predicted Probability Distribution -------------------------------------
prob_df <- data.frame(
  Probability = nn_pred_prob,
  Actual      = factor(test_nn$Bankrupt, levels = c(0, 1),
                        labels = c("Not Bankrupt", "Bankrupt"))
)
 
p_prob <- ggplot(prob_df, aes(x = Probability, fill = Actual)) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("Not Bankrupt" = "#2ECC71",
                               "Bankrupt"     = "#E74C3C")) +
  labs(
    title    = "Neural Network — Predicted Probability Distribution",
    subtitle = "Distribution of predicted bankruptcy probabilities by actual class",
    x = "Predicted Probability of Bankruptcy",
    y = "Count", fill = ""
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")
 
ggsave("plot_18_nn_prob_distribution.png",
       plot = p_prob, width = 8, height = 5, dpi = 150)
cat(">> Saved: plot_18_nn_prob_distribution.png\n")
 
# --- Network Architecture Diagram -------------------------------------------
# Plot the neural network structure
png("plot_19_nn_architecture.png", width = 1400, height = 900, res = 120)
plot(nn_model, rep = "best",
     main = "Neural Network Architecture\n(Input -> 16 nodes -> 8 nodes -> Output)")
dev.off()
cat(">> Saved: plot_19_nn_architecture.png\n")
 
# =============================================================================
# STEP 7: SAVE RESULTS FOR MODEL COMPARISON
# =============================================================================
 
cat("\n=============================================\n")
cat("STEP 7: SAVE RESULTS\n")
cat("=============================================\n")
 
nn_results <- list(
  model   = nn_model,
  cm      = cm,
  roc     = roc_obj,
  auc     = auc_val,
  metrics = data.frame(
    Model       = "Neural Network",
    Accuracy    = accuracy,
    Precision   = precision,
    Recall      = recall,
    F1_Score    = f1,
    Specificity = specificity,
    AUC_ROC     = auc_val
  )
)
 
saveRDS(nn_results, "nn_results.rds")
cat(">> Results saved to: nn_results.rds\n")
cat("   (Used in model comparison script)\n")
 
# =============================================================================
# SUMMARY
# =============================================================================
 
cat("\n=============================================\n")
cat("NEURAL NETWORK — SUMMARY\n")
cat("=============================================\n")
cat("Architecture: Input ->  16 -> 8 -> Output\n")
cat("Activation  : Logistic (sigmoid)\n")
cat(sprintf("Accuracy    : %.2f%%\n", accuracy))
cat(sprintf("Precision   : %.2f%%\n", precision))
cat(sprintf("Recall      : %.2f%%\n", recall))
cat(sprintf("F1 Score    : %.2f%%\n", f1))
cat(sprintf("Specificity : %.2f%%\n", specificity))
cat(sprintf("AUC-ROC     : %.4f\n",   auc_val))
cat("\nOutput files:\n")
cat("  plot_16_nn_confusion_matrix.png\n")
cat("  plot_17_nn_roc_curve.png\n")
cat("  plot_18_nn_prob_distribution.png\n")
cat("  plot_19_nn_architecture.png\n")
cat("  nn_results.rds\n")
cat("=============================================\n")