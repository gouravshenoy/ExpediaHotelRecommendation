import random
from DecisionTree.DecisionTree.DecisionTree import DecisionTree


class EnsembleUtil:
    """
    This is a utility class for performing Ensemble learning such as Bagging/Boosting
    """

    def createBootstrapSamples(self, trainingSet, numBags):
        """
        This function creates 'numBags' bootstrap samples (with replacement)
        randomly drawn from the trainingSet
        :param trainingSet: the training data
        :param numBags: number of bootstrapped bags
        :return: bootstrapped samples
        """
        bootstrap_samples = []
        sample_size = len(trainingSet)

        for index in range(numBags):
            # print("Creating bootstrap sample #" + str(index))
            bootstrap_samples.append(
                [random.choice(trainingSet) for _ in range(sample_size)]
            )

        return bootstrap_samples

    def get_majority_voted_labels(self, predicted_label_collection):
        """
        This function uses majority voting to return the class label
        from list of labels predicted by each learned model
        :param predicted_label_collection: labels predicted by learned model(s)
        :return: majority voted class labels
        """
        num_bags = len(predicted_label_collection)
        num_test_points = len(predicted_label_collection[0])

        # labels selected via majority vote - appearing more times
        majority_voted_labels = []

        for test_data_index in range(num_test_points):
            # keep count of majority vote
            label_count = {True.__int__(): 0,
                           False.__int__(): 0}

            # scan through all bags & increase vote
            for bag_index in range(num_bags):
                label_count[
                    predicted_label_collection[bag_index][test_data_index]
                ] += 1

            # add majority voted class-label
            majority_voted_labels.append(
                (label_count[True.__int__()] >= label_count[False.__int__()]).__int__()
            )

        # return majority vote
        return majority_voted_labels

    def calculate_accuracy(self, testing_data, predicted_classes):
        """
        This function prints the accuracy & mis-classification count
        of the ensemble method
        :param testing_data: test data set
        :param predicted_classes: predicted classes by ensemble learner
        :return:
        """
        decision_tree = DecisionTree()

        # calculate accuracy of model
        dt_accuracy, dt_misclassification = decision_tree.calculateAccuracy(testing_data,
                                                                            predicted_classes)

        print('Accuracy of ensemble method = {}%'.format(round(dt_accuracy, 3)))
        print('Misclassification Count = {}'.format(dt_misclassification))
        return


    def print_confusion_matrix(self, testing_data, predicted_classes):
        """
        This function prints the confusion matrix for the ensemble learner
        :param testing_data: test data set
        :param predicted_classes: predicted classes by ensemble learner
        :return:
        """
        decision_tree = DecisionTree()

        # print confusion matrix
        decision_tree.plotConfusionMatrix(testing_data,
                                          predicted_classes)
        return
