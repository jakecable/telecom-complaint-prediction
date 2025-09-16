library(ggplot2)
library(dplyr)

# Set the working directory to the location of the script
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

data <- read.csv("../data/processed/unified_model_dataset_corrected.csv")


# Scatter plot of Total Population vs. Complaint Volume
# This explores the relationship between population and complaints

scatter_population <- ggplot(data, aes(x = total_population, y = complaint_volume)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  ggtitle("Complaint Volume vs. Total Population")+
  xlab("Total Population")+
  ylab("Complaint Volume")

ggsave("../outputs/scatter_population_vs_complaints.png", plot = scatter_population)

# Linear regression model built to predict 'complaint_volume' based on the other numeric variables
# Exclude non-predictor columns like zip_code and state
model <- lm(complaint_volume ~ total_population + median_age + total_housing_units + avg_pct_broadband_25_3 + avg_pct_broadband_100_20, data = data)

print(summary(model))