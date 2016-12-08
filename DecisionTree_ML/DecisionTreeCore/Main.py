import os
from DecisionTreeCore import Data
from DecisionTreeCore import Constants
from DecisionTreeCore import InputHandler
from DecisionTreeCore import DecisionTree
from DecisionTreeCore import ResultPlotter

# import used for creating csv file
import csv

from pprint import pprint

class Main:
    """
    Main class the entry point for the program
    """

    def run_dt_algo(self):
        # create new data & input-handler object
        data_train = Data.Data()
        data_test = Data.Data()
        inputHandler = InputHandler.InputHandler()

        # create dict for accuracy at depth
        accuracy_dict = {}

        # create data matrix from training file
        print ("Reading Training File & Creating data-matrix")
        matrix_train = inputHandler.readFile(Constants.TRAIN_FILE_NAME,
                                             Constants.LABEL_INDEX,
                                             Constants.FEATURE_INDICES,
                                             fileType="csv",
                                             ignoreHeader=True)

        # create data matrix from testing file
        print ("Reading Testing File & Creating data-matrix")
        matrix_test = inputHandler.readFile(Constants.TEST_FILE_NAME,
                                            Constants.LABEL_INDEX,
                                            Constants.FEATURE_INDICES,
                                            fileType="csv",
                                            ignoreHeader=True)

        # set the data matrices in the object
        data_train.setMatrix(matrix_train)
        data_test.setMatrix(matrix_test)

        # create decision-tree object
        decisionTree = DecisionTree.DecisionTree()
        print ('{0:{fill}{align}{width}} '.format('=',
                                                  fill='=',
                                                  width=65,
                                                  align='^'))
        print ("-- Learning decision tree from file = {} ".format(os.path.basename(Constants.TRAIN_FILE_NAME)))
        print ('{0:{fill}{align}{width}} '.format('=',
                                                  fill='=',
                                                  width=65,
                                                  align='^'))

        # run for max-depth times
        # for tree_depth in range(1, Constants.TREE_DEPTH):
        tree_depth = Constants.TREE_DEPTH
        print ("\n- Tree Depth = {}".format(tree_depth))

        ''' ~~ Training Phase ~~ '''
        # run the decision-tree training algorithm
        decisionTree.train(data_train, treeDepth=tree_depth)

        # optional - print the generated tree
        '''
            Uncomment this line to get a visualization of
            the decision tree generated. You need to have
            pydot python library installed for this to work.
        '''
        # decisionTree.printTree(op_file=os.path.basename(Constants.TRAIN_FILE_NAME))

        ''' ~~ Testing Phase ~~ '''
        # get prediction accuracy from trained model
        predicted_classes = decisionTree.test(data_test.getMatrix())

        # calculate accuracy of model
        dt_accuracy, dt_misclassification = decisionTree.calculateAccuracy(data_test.getMatrix(),
                                                                           predicted_classes)
        print('- Accuracy of model = {}'.format(dt_accuracy))
        print('- Misclassification Count = {}'.format(dt_misclassification))
        accuracy_dict[tree_depth] = dt_accuracy

        decisionTree.plotConfusionMatrix(data_test.getMatrix(),
                                         predicted_classes)

    def plot_accuracy_graph(self):
        """
        This function invokes the plotting module in the program. It will plot the Accuracy vs Tree Depth
        for all 3 monk data sets.
        :return: none
        """
        # csv library is used only for writing the csv file
        #   csv file is used for plotting graph
        with open(Constants.RESULT_FILE, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Depth of Tree', 'Monk-1 Accuracy', 'Monk-2 Accuracy', 'Monk-3 Accuracy', 'Avg. Accuracy'])
            for depth, accuracy in self.__avg_accuracies.items():
                writer.writerow([depth,
                                 self.__monk_accuracies[0][depth],
                                 self.__monk_accuracies[1][depth],
                                 self.__monk_accuracies[2][depth], accuracy])

        ResultPlotter.plotResults()
        pass


# run main function
if __name__ == '__main__':
    main_obj = Main()

    # run DT algorithm on expedia data-set
    main_obj.run_dt_algo()

    # run DT algorithm on Monks data-set
    # main_obj.run_monk()

    # plot accuracay graph for Monks data-set
    # main_obj.plot_accuracy_graph()
