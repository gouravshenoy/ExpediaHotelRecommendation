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

- We have placed the __monks__ training & testing data files (3 files each), as well as our own data-set's training & testing files _(mammographic-masses)_ in the `data` directory. The code has been updated to point to this location.

- We have set max depth of the tree = 10. This can be changed by updating the `Constants.py` file, replacing the _TREE_DEPTH_ field.

- Printing of decision tree (visualization) has been disabled in the code. We have pre-plotted the trees for each of the monks data-set & our own data-set for depths 1,2 and placed them in the `dt-plots` directory. 

## Running the program

- We have created a requirements file to make installing dependencies easy. Install using:
`pip install -r requirements.txt`

- Run the program with the following command:
`python Main.py`

## Program Output

- The program should output accuracy of the model, mis-classification count for all data-sets for depths (1..9). 

- The program also output's the Confusion Matrix for depth 1,2 for all data-sets.

- Finally, the program will display a graph of accuracy vs tree-depth for all 3 Monk data-sets as well as the average accuracy of the 3 data-sets. 

```
--- DECISION TREE ALGORITHM (MONKS DATA-SET) --- 
================================================================= 
-- Learning decision tree from file = monks-1.train.txt 
================================================================= 

- Tree Depth = 1
- Accuracy of model = 75.0
- Misclassification Count = 108

- Confusion Matrix:
                            predicted label                  
----------------------------------------------------------------- 
                        | label = 0  | label = 1  | 
----------------------------------------------------------------- 
           | label = 0  |  TN: 216   |   FP: 0    | Total = 216 | 
true label ------------------------------------------------------
           | label = 1  |  FN: 108   |  TP: 108   | Total = 216 | 
----------------------------------------------------------------- 
                         Total = 324 | Total = 108 |    432     | 
----------------------------------------------------------------- 

- Tree Depth = 2
- Accuracy of model = 72.2222222222
- Misclassification Count = 120

- Confusion Matrix:
                            predicted label                  
----------------------------------------------------------------- 
                        | label = 0  | label = 1  | 
----------------------------------------------------------------- 
           | label = 0  |  TN: 192   |   FP: 24   | Total = 216 | 
true label ------------------------------------------------------
           | label = 1  |   FN: 96   |  TP: 120   | Total = 216 | 
----------------------------------------------------------------- 
                         Total = 288 | Total = 144 |    432     | 
----------------------------------------------------------------- 

- Tree Depth = 3
- Accuracy of model = 75.2314814815
- Misclassification Count = 107

- Tree Depth = 4
- Accuracy of model = 77.3148148148
- Misclassification Count = 98

- Tree Depth = 5
- Accuracy of model = 77.5462962963
- Misclassification Count = 97

- Tree Depth = 6
- Accuracy of model = 77.5462962963
- Misclassification Count = 97

- Tree Depth = 7
- Accuracy of model = 77.5462962963
- Misclassification Count = 97

- Tree Depth = 8
- Accuracy of model = 77.5462962963
- Misclassification Count = 97

- Tree Depth = 9
- Accuracy of model = 77.5462962963
- Misclassification Count = 97
================================================================= 
-- Learning decision tree from file = monks-2.train.txt 
================================================================= 

- Tree Depth = 1
- Accuracy of model = 67.1296296296
- Misclassification Count = 142

- Confusion Matrix:
                            predicted label                  
----------------------------------------------------------------- 
                        | label = 0  | label = 1  | 
----------------------------------------------------------------- 
           | label = 0  |  TN: 290   |   FP: 0    | Total = 290 | 
true label ------------------------------------------------------
           | label = 1  |  FN: 142   |   TP: 0    | Total = 142 | 
----------------------------------------------------------------- 
                         Total = 432 | Total = 0  |    432     | 
----------------------------------------------------------------- 

- Tree Depth = 2
- Accuracy of model = 60.6481481481
- Misclassification Count = 170

- Confusion Matrix:
                            predicted label                  
----------------------------------------------------------------- 
                        | label = 0  | label = 1  | 
----------------------------------------------------------------- 
           | label = 0  |  TN: 222   |   FP: 68   | Total = 290 | 
true label ------------------------------------------------------
           | label = 1  |  FN: 102   |   TP: 40   | Total = 142 | 
----------------------------------------------------------------- 
                         Total = 324 | Total = 108 |    432     | 
----------------------------------------------------------------- 

- Tree Depth = 3
- Accuracy of model = 63.1944444444
- Misclassification Count = 159

- Tree Depth = 4
- Accuracy of model = 63.8888888889
- Misclassification Count = 156

- Tree Depth = 5
- Accuracy of model = 66.2037037037
- Misclassification Count = 146

- Tree Depth = 6
- Accuracy of model = 66.2037037037
- Misclassification Count = 146

- Tree Depth = 7
- Accuracy of model = 66.2037037037
- Misclassification Count = 146

- Tree Depth = 8
- Accuracy of model = 66.2037037037
- Misclassification Count = 146

- Tree Depth = 9
- Accuracy of model = 66.2037037037
- Misclassification Count = 146
================================================================= 
-- Learning decision tree from file = monks-3.train.txt 
================================================================= 

- Tree Depth = 1
- Accuracy of model = 80.5555555556
- Misclassification Count = 84

- Confusion Matrix:
                            predicted label                  
----------------------------------------------------------------- 
                        | label = 0  | label = 1  | 
----------------------------------------------------------------- 
           | label = 0  |  TN: 132   |   FP: 72   | Total = 204 | 
true label ------------------------------------------------------
           | label = 1  |   FN: 12   |  TP: 216   | Total = 228 | 
----------------------------------------------------------------- 
                         Total = 144 | Total = 288 |    432     | 
----------------------------------------------------------------- 

- Tree Depth = 2
- Accuracy of model = 97.2222222222
- Misclassification Count = 12

- Confusion Matrix:
                            predicted label                  
----------------------------------------------------------------- 
                        | label = 0  | label = 1  | 
----------------------------------------------------------------- 
           | label = 0  |  TN: 204   |   FP: 0    | Total = 204 | 
true label ------------------------------------------------------
           | label = 1  |   FN: 12   |  TP: 216   | Total = 228 | 
----------------------------------------------------------------- 
                         Total = 216 | Total = 216 |    432     | 
----------------------------------------------------------------- 

- Tree Depth = 3
- Accuracy of model = 97.2222222222
- Misclassification Count = 12

- Tree Depth = 4
- Accuracy of model = 93.0555555556
- Misclassification Count = 30

- Tree Depth = 5
- Accuracy of model = 93.9814814815
- Misclassification Count = 26

- Tree Depth = 6
- Accuracy of model = 93.9814814815
- Misclassification Count = 26

- Tree Depth = 7
- Accuracy of model = 93.9814814815
- Misclassification Count = 26

- Tree Depth = 8
- Accuracy of model = 93.9814814815
- Misclassification Count = 26

- Tree Depth = 9
- Accuracy of model = 93.9814814815
- Misclassification Count = 26

--- DECISION TREE ALGORITHM (OWN DATA-SET) --- 
================================================================= 
-- Learning decision tree from file = mammographic-masses-train.txt 
================================================================= 

- Tree Depth = 1
Saving decision tree visualization in dt_plots folder!
- Accuracy of model = 74.2268041237
- Misclassification Count = 25

- Confusion Matrix:
                            predicted label                  
----------------------------------------------------------------- 
                        | label = 0  | label = 1  | 
----------------------------------------------------------------- 
           | label = 0  |   TN: 72   |   FP: 0    | Total = 72 | 
true label ------------------------------------------------------
           | label = 1  |   FN: 25   |   TP: 0    | Total = 25 | 
----------------------------------------------------------------- 
                         Total = 97 | Total = 0  |     97     | 
----------------------------------------------------------------- 

- Tree Depth = 2
Saving decision tree visualization in dt_plots folder!
- Accuracy of model = 82.4742268041
- Misclassification Count = 17

- Confusion Matrix:
                            predicted label                  
----------------------------------------------------------------- 
                        | label = 0  | label = 1  | 
----------------------------------------------------------------- 
           | label = 0  |   TN: 60   |   FP: 12   | Total = 72 | 
true label ------------------------------------------------------
           | label = 1  |   FN: 5    |   TP: 20   | Total = 25 | 
----------------------------------------------------------------- 
                         Total = 65 | Total = 32 |     97     | 
----------------------------------------------------------------- 

- Tree Depth = 3
- Accuracy of model = 79.381443299
- Misclassification Count = 20

- Tree Depth = 4
- Accuracy of model = 78.3505154639
- Misclassification Count = 21

- Tree Depth = 5
- Accuracy of model = 78.3505154639
- Misclassification Count = 21

- Tree Depth = 6
- Accuracy of model = 78.3505154639
- Misclassification Count = 21

- Tree Depth = 7
- Accuracy of model = 78.3505154639
- Misclassification Count = 21
```

- The accuracy vs tree depth plot looks as follows:

![plot](dt-plots/accuracy_vs_depth_plot.png)