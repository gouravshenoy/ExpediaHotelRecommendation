from DecisionTree.BaggingBoosting.BaggingEnsemble import BaggingEnsemble
from DecisionTree.BaggingBoosting.EnsembleUtil import EnsembleUtil

from pprint import pprint

from DecisionTree.BaggingBoosting.AdaBoostClassifier import AdaBoostClassifier

class Main:
    """
    Main class with implementation logic interfaces for Bagging & Boosting
    """

    def __init__(self, matrix_train, matrix_test, tree_depth, num_bags):
        """
        Constructor for this class
        :param matrix_train: train data set
        :param matrix_test: test data set
        :param tree_depth: depth of DT
        :param num_bags: number of bootstrapped bags
        """
        self.__matrix_train = matrix_train
        self.__matrix_test = matrix_test
        self.__tree_depth = tree_depth
        self.__num_bags = num_bags
        return


    def run_bagging(self):
        """
        This function runs the bagging ensemble algorithm. It learns 'self.__num_bags' DTs of depth 'self.__tree_depth',
        and uses majority voting to predict the class label for the test data.
        :return:
        """
        ensemble_util = EnsembleUtil()

        """ Commenting out because load_data() function in pa2_template.py
            performs the same logic, and this run_bagging() method is called
            in the learn_bagged() method in pa2_template.py

        inputHandler = InputHandler()

        # create data matrix from training file
        matrix_train = inputHandler.readFile(Constants.TRAIN_FILE_NAME,
                                             Constants.LABEL_INDEX,
                                             fileType="csv",
                                             ignoreHeader=True)

        # create data matrix from testing file
        matrix_test = inputHandler.readFile(Constants.TEST_FILE_NAME,
                                            Constants.LABEL_INDEX,
                                            fileType="csv",
                                            ignoreHeader=True)

        print("Number of training examples: " + str(self.__matrix_train.__len__()))
        print("Number of testing examples: " + str(self.__matrix_test.__len__()))
        """

        # initialize bagging ensemble
        bagging_ensemble = BaggingEnsemble(training_data=self.__matrix_train,
                                           num_bags=self.__num_bags)

        # learn the bagging ensemble
        bagging_ensemble.learn(self.__tree_depth)

        # predict using bagging
        predicted_classes = bagging_ensemble.predict(testing_data=self.__matrix_test)
        # print(predicted_classes)

        # print accuracy & misclassification count
        print("\nTASK: Printing Bagging accuracy & mis-classification count")
        ensemble_util.calculate_accuracy(testing_data=self.__matrix_test,
                                         predicted_classes=predicted_classes)

        # print confusion matrix
        print("\nTASK: Printing Confusion Matrix")
        ensemble_util.print_confusion_matrix(testing_data=self.__matrix_test,
                                             predicted_classes=predicted_classes)

        return

    def run_boosting(self):

        ensemble_util = EnsembleUtil()

        # initialize boosting ensemble
        adaBoost_ensemble = AdaBoostClassifier(training_data=self.__matrix_train,
                                               tree_depth=self.__tree_depth,
                                               iteration_count=self.__num_bags)

        # learn the boosting ensemble
        adaBoost_ensemble.learn();

        # predict using boosting
        predicted_classes = adaBoost_ensemble.test(testing_data=self.__matrix_test)
        # print(predicted_classes)

        # print accuracy & misclassification count
        print("\nTASK: Print AdaBoost accuracy & mis-classification count")
        ensemble_util.calculate_accuracy(testing_data=self.__matrix_test,
                                         predicted_classes=predicted_classes)

        # print confusion matrix
        print("\nTASK: Print Confusion Matrix")
        ensemble_util.print_confusion_matrix(testing_data=self.__matrix_test,
                                             predicted_classes=predicted_classes)


# run main function
# if __name__ == '__main__':
#     main_obj = Main()
#
#     # dry run
#     main_obj.run_bagging()
