## This file holds the constants and input parameters to the program

# Training data set
TRAIN_FILE_NAME = "/Users/goshenoy/SOIC-Courses/Applied-ML/Final_Project/Datasets/cleaned_and_merged/train_100k_sample_cleaned.csv"

# Test data set
TEST_FILE_NAME = "/Users/goshenoy/SOIC-Courses/Applied-ML/Final_Project/Datasets/cleaned_and_merged/train_10k_sample_cleaned.csv"

# Output result file from the program
RESULT_FILE = "dt_accuracies.csv"

# Index of the class label in expedia data
LABEL_INDEX = 175

# Indices of features in the expedia data
FEATURE_INDICES = (0,174)

# Desired max depth of the tree
TREE_DEPTH = 20

# Number of distinct values of the class label
FEATURE_DIMENSION = 100