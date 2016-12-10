# DecisionTree

## Pre-requisite libraries

This project has been implemented in Python, and python version required is __3.5.2__.

Additional python packages have used solely for the purpose of visualization. These are:

- pandas - This package is used for plotting the accuracy vs depth graph. You will need to install this package using:
`pip install pandas`

- matplotlib - This package is used for plotting the accuracy vs depth graph. You will need to install this package using:
`pip install matplotlib`

- pydot - This package is used for drawing the decision tree. Install this package using:
`pip install pydot`

- uuid - This package is used for drawing the decision tree. This package comes pre-installed with python-3.5.2 distribution.

## Pre-requisite configuration

Before you can run the program, you need to ensure the following:

- Update the location of the train and test data-sets in the `Constants.py` file, replacing the _TRAIN_FILE_NAME_ & _TEST_FILE_NAME_ fields.

- We have set max depth of the tree = 10. This can be changed by updating the `Constants.py` file, replacing the _TREE_DEPTH_ field.

- Printing of decision tree (visualization) has been disabled in the code. We have pre-plotted the trees for each of the monks data-set & our own data-set for depths 1,2 and placed them in the `dt-plots` directory. 

## Running the program

- We have created a requirements file to make installing dependencies easy. Install using:
`pip install -r requirements.txt`

- Run the program with the following command:
`python Main.py`
