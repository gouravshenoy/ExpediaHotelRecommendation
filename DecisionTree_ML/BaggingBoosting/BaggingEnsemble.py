from DecisionTree.BaggingBoosting.EnsembleUtil  import EnsembleUtil
from DecisionTree.DecisionTree.Data import Data
from DecisionTree.DecisionTree.DecisionTree import DecisionTree

from pprint import pprint
import copy

class BaggingEnsemble:
    """
    This class contains core implementation for Bagging ensemble learner.
    """

    def __init__(self, training_data, num_bags):
        """
        Constructor for this class
        :param training_data: the training data
        :param num_bags: number of bootstrapped bags
        """
        self.__training_data = training_data
        self.__num_bags = num_bags
        self.__bootstrap_samples = []
        self.__learned_models = []
        self.__predicted_classes_collection = []


    def learn(self, tree_depth):
        """
        This functions learns 'self.__num_bags' decision-trees of 'tree_depth' height each.
        :param tree_depth: depth of DT
        :return:
        """
        ensembleUtil = EnsembleUtil()

        # get bootstrap samples
        print("\nTASK: Creating Bootstrap Samples from Training Data")
        self.__bootstrap_samples = ensembleUtil.createBootstrapSamples(self.__training_data,
                                                                       self.__num_bags)

        print("\nTASK: Learning Decision Tree for each Bootstrap Sample")
        # for each bootstrap sample, learn a DT
        for count, bootstrap_sample in enumerate(self.__bootstrap_samples):
            # print("Learning DT for bootstrapped sample #" + str(count))
            data_train = Data()
            decision_tree = DecisionTree()

            # run the decision-tree training algorithm
            data_train.setMatrix(bootstrap_sample)
            decision_tree.train(data=data_train,
                                treeDepth=tree_depth)

            # decision_tree.printTree('bagging-' + str(count))

            # save the learned model
            self.__learned_models.append(copy.deepcopy(decision_tree))
        return


    def predict(self, testing_data):
        """
        This function predicts the class labels for the testing data set
        using majority voting
        :param testing_data: the testing data set
        :return: predicted class labels
        """
        ensembleUtil = EnsembleUtil()

        # predict using each DT model
        print("\nTASK: Getting predicted class label by each learned DT model")

        # create test data matrix
        data_test = Data()
        data_test.setMatrix(testing_data)

        for count, dt_model in enumerate(self.__learned_models):
            # print("Predicting class labels with DT model #" + str(count))
            self.__predicted_classes_collection.append(
                dt_model.test(data_test.getMatrix())
            )

        #print(self.__predicted_classes_collection)
        print("\nTASK: Selecting final class labels using majority voting")
        return ensembleUtil.get_majority_voted_labels(self.__predicted_classes_collection)