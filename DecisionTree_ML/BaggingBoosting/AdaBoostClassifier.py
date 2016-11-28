from DecisionTree.DecisionTree.Data import Data
from DecisionTree.DecisionTree.WeightedDecisionTree import WeightedDecisionTree
import math
import copy

class AdaBoostClassifier:

    def __init__(self, training_data, tree_depth, iteration_count):
        '''
        This function initializes the Adaboost with specified parameters.
        :param training_data: training data matrix
        :param tree_depth: depth of the tree
        :param iteration_count: number of iteration for boosting
        '''
        self.__training_data = training_data
        self.__tree_depth = tree_depth
        self.__iteration_count = iteration_count
        self.__learned_models = []
        self.__learned_models_weight = []

    def learn(self):
        '''
        This function learn the adaboost ensemble method with specified parameters.
        :return:
        '''
        n = len(self.__training_data)
        D = [1/n for i in range(0,n)]
        trainData = Data()
        trainData.setMatrix(self.__training_data)
        column_count = len(self.__training_data[0])

        for index in range(0, self.__iteration_count):
            misclassified_examples = []
            # learn decision tree model with specified depth
            model = WeightedDecisionTree()
            model.train(trainData, D, self.__tree_depth)
            train_predicted_labels  = model.test(trainData.getMatrix())
            for i in range(0, len(self.__training_data)):
                actual_label = self.__training_data[i][column_count-1]
                if(actual_label != train_predicted_labels[i]):
                    misclassified_examples.append(i)

            alpha, D = self.calculate_updated_weights(misclassified_examples, D)

            # Add model to learned model list
            self.__learned_models.append(copy.deepcopy(model))
            self.__learned_models_weight.append(alpha)



    def calculate_updated_weights(self, misclassified_examples, weights):
        '''
        This function calculate new weights and alpha for the model.
        :param misclassified_examples: list mis-classified exmaples index
        :param weights: current weights
        :return: Updated weights and alpha for the model
        '''
        error = 0.00000000000001 # epsilon
        for index in misclassified_examples:
            error = error + weights[index]

        alpha = 0.5 * math.log(float(1-error)/error, math.e)

        updated_weight_sum = 0
        for index in range(0, len(self.__training_data)):
            if index in misclassified_examples:
                weights[index] = weights[index] * (math.pow(math.e, alpha))
            else:
                weights[index] = weights[index] * math.pow(math.e, (-1*alpha))
            updated_weight_sum = updated_weight_sum + weights[index]

        for index in range(0, len(self.__training_data)):
            weights[index] = weights[index]/updated_weight_sum

        return alpha, weights

    def test(self, testing_data):
        '''
        This function runs learned adaboost ensemble model on specified testing data
        :param testing_data: testing data
        :return: list of predicted labels for the test data
        '''
        data = Data()
        data.setMatrix(testing_data)
        cumulative_predicted_labels = {}
        for index in range(0,len(testing_data)):
            cumulative_predicted_labels[index] = 0

        for model_index in range(0, len(self.__learned_models)):
            model = self.__learned_models[model_index]
            model_weight = self.__learned_models_weight[model_index]
            predicted_labels = model.test(testing_data)
            for index, predicted_label in enumerate(predicted_labels):
                if predicted_label == 0:
                    cumulative_predicted_labels[index] -= model_weight
                else:
                    cumulative_predicted_labels[index] += model_weight

        for index in range(0, len(cumulative_predicted_labels)):
            if cumulative_predicted_labels[index] < 0:
                cumulative_predicted_labels[index] = 0
            else:
                cumulative_predicted_labels[index] = 1

        return cumulative_predicted_labels
