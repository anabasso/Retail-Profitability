# LIBRARIES ----
library(reshape2) # for melting the correlation matrix
library(ggplot2) # for plotting the correlation matrix
library(e1071) # for svm model
library(randomForest)
library(plyr)
library(dplyr)

# FUNCTIONS ----
# Error metrics
mae <- function(error){
  return(mean(abs(error)))
}
rmse <- function(error){
  return(sqrt(mean(error^2)))
}
r2 <- function(error, reference){
  return(1 - (sum(error^2) / sum((reference - mean(reference))^2)))
}


# DATA ----
dt_raw <- read.csv("existing-products.csv", na.strings = "?")
str(dt_raw)
dt_raw$ProductID <- factor(dt_raw$ProductID)

# Explore
head(dt_raw)
tail(dt_raw) 
summary(dt_raw)

# Find NAs
which(is.na(dt_raw))
1 + which(is.na(dt_raw)) / nrow(dt_raw)
colnames(dt_raw[12])

# Clean unnecessary attributes
dt <- dt_raw
dt[, c("BestSellersRank", "ShippingWeight", "ProductDepth", "ProductWidth", "ProductHeight")] <- NULL

# Data distribution
summary(dt)
boxplot(dt$Volume)

# Identify and eliminate outliers
which(dt$Volume > 4000)
dt <- dt[-which(dt$Volume > 4000),]

# LINEAR MODEL ----
trainSize <- round(nrow(dt) * 0.7)
set.seed(123)
training_indices <- sample(seq_len(nrow(dt)), size = trainSize)
trainSet <- dt[training_indices, ]
testSet <- dt[-training_indices, ]

lm_fit <- lm(formula = Volume ~ . , data = trainSet[,-c(1,2)])
summary(lm_fit)
plot(lm_fit)

testSet$lm_predict <- predict(lm_fit, testSet) #, interval = "predict", level = .95)
testSet$lm_error <- testSet$lm_predict - testSet$Volume
boxplot(testSet$lm_error)
summary(testSet$lm_error)

# Calculate MAE, RMSE and R Squared
mae(testSet$lm_error)
rmse(testSet$lm_error)
r2(testSet$lm_error, testSet$Volume)

# CORRELATION MATRIX ----
# Tutorial for creating a correlation matrix with heatmap and values
# from http://www.sthda.com/english/wiki/ggplot2-quick-correlation-matrix-heatmap-r-software-and-data-visualization

# OPTION 1 - Simplified
cormat <- round(cor(dt[-c(1, 2)]), 2)
melted_cormat <- melt(cormat)
ggplot(melted_cormat, aes(Var1, Var2, fill = value)) +
  geom_tile()

# OPTION 2 - For getting just the upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)] <- NA
  return(cormat)
}
upper_tri <- get_upper_tri(cormat)
melted_cormat <- melt(upper_tri, na.rm = T)
ggplot(melted_cormat, aes(Var2, Var1, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name = "Pearson\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()

# OPTION 3 - Reorder the attributes, heatmap and values
reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1 - cormat) / 2)
  hc <- hclust(dd)
  cormat <- cormat[hc$order, hc$order]
}
cormat <- reorder_cormat(cormat)
upper_tri <- get_upper_tri(cormat)
melted_cormat <- melt(upper_tri, na.rm = T)
ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name = "Pearson\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()

# Add correlation coefficients on the heatmap
ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))

# FEATURE SELECTION ----
features <- which(colnames(dt) %in% c("Price", "FourStarReviews", "PositiveServiceReview", "NegativeServiceReview", "WouldConsumerRecommend", "Volume"))

# NEW MODELS ----
# Linear Model ----
lm2_fit <- lm(formula = Volume ~ . , 
             data = trainSet[,features])
summary(lm2_fit)

testSet$lm2_predict <- predict(lm2_fit, testSet)
testSet$lm2_error <- testSet$lm2_predict - testSet$Volume
boxplot(testSet$lm2_error)
summary(testSet$lm2_error)

# Support Vector Machine ----
svm_fit <- svm(formula = Volume ~ . , 
             data = trainSet[,features])
summary(svm_fit)

testSet$svm_predict <- predict(svm_fit, testSet)
testSet$svm_error <- testSet$svm_predict - testSet$Volume
boxplot(testSet$svm_error)
summary(testSet$svm_error)

# Random Forest ----
rf_fit <- randomForest(formula = Volume ~ . , 
               data = trainSet[,features])
summary(rf_fit)

testSet$rf_predict <- predict(rf_fit, testSet)
testSet$rf_error <- testSet$rf_predict - testSet$Volume
boxplot(testSet$rf_error)
summary(testSet$rf_error)

# Compare models ----
model <- c("lm", "svm", "rf")
comp_mae <- c(mae(testSet$lm2_error), 
              mae(testSet$svm_error), 
              mae(testSet$rf_error))
comp_rmse <- c(rmse(testSet$lm2_error), 
               rmse(testSet$svm_error), 
               rmse(testSet$rf_error))
comp_r2 <- c(r2(testSet$lm2_error, testSet$Volume), 
             r2(testSet$svm_error, testSet$Volume), 
             r2(testSet$rf_error, testSet$Volume))

compare <- data.frame(model, comp_mae, comp_rmse, comp_r2)
ggplot(compare, aes(model, comp_mae, group = 1)) +
  geom_line()

ggplot(compare, aes(model, comp_rmse, group = 1)) +
  geom_line()

ggplot(compare, aes(model, comp_r2, group = 1)) +
  geom_line()

boxplot(testSet$lm2_error, testSet$svm_error, testSet$rf_error)
summary(testSet[c("lm2_error", "svm_error", "rf_error")])

# PREDICT PROFITABILITY ----
# Products to predict
dt_predict <- read.csv("potential-products.csv", na.strings = "?")
str(dt_predict)
dt_predict$ProductID <- factor(dt_predict$ProductID)

# Explore
head(dt_predict)
tail(dt_predict) 
summary(dt_predict)

# Find NAs
which(is.na(dt_predict))
1 + which(is.na(dt_predict)) / nrow(dt_predict)
colnames(dt_raw[18])

# Predict
dt_predict$volume_predict <- predict(rf_fit, dt_predict)
dt_predict$profit_predict <- dt_predict$volume_predict * dt_predict$ProfitMargin * dt_predict$Price

head(dt_predict %>% 
  arrange(desc(profit_predict)) %>% 
  select(ProductType,
         ProductID,
         Price,
         volume_predict,
         profit_predict), 10)
