## Naive Bayes Implementation

### Prerequisites
- python 3.5.2
- python pandas library

### Configuration
Update the _train_file_ and _test_file_ variables in `naive_bayes.py` file to point to the train and test datasets respectively. 

### How to run
```
$ python naive_bayes.py
```

### Cleaning & Merging datasets
This program performs the discretization of the train & test datasets. It also merges the train data with `destinations.csv` data set. 
Update the _datasets_ variable to point to the datafiles that need to be cleaned and merged, and the _op_files_ variable to point to the output file location.
To run the program execute the following:
```
$ python clean_dataset.py
```
