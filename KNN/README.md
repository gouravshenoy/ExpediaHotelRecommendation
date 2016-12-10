## KNN Implementation

### Prerequisites
- python 2.7.12
- python pandas library

### Preprocessing
From the dataset csv (test/train), delete all the date columns. The date columns are non-numeric and hence skipped.

### How to run
```
$python expedia_knn.py <train_csv_path> <test_csv_path> [<K Value>]
$python expedia_knn_with_mean.py <train_csv_path> <test_csv_path> [<K Value>]
$python expedia_knn_with_sampling.py <train_csv_path> <test_csv_path> [<K Value>] [<Number of Bootstraps]
```
