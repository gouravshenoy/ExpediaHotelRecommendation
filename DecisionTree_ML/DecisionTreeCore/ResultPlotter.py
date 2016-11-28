"""
Matplotlib and Pandas are used the draw the accuracy vs tree-depth plots.
"""
import matplotlib.pyplot as plt
import pandas as pd
from DecisionTreeCore import Constants

def plotResults():
    """
    This function draws the plot of Accuracy vs Tree Depth.
    :return: None
    """
    result = pd.read_csv(Constants.RESULT_FILE)
    #print("Test RESULT", result)


    plt.plot(result[[0]], result[[1]], 'blue', linewidth='1', label='Monk-1')
    plt.plot(result[[0]], result[[2]], 'green', linewidth='1', label='Monk-2')
    plt.plot(result[[0]], result[[3]], 'red', linewidth='1', label='Monk-3')
    plt.plot(result[[0]], result[[4]], 'black', linewidth='3', label='Average')

    plt.ylabel('Accuracy')
    plt.xlabel('Depth of Tree')
    plt.legend(loc='lower right', shadow=True)
    plt.show()