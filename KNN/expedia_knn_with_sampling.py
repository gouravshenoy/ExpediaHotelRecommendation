__author__ = "Mangirish Wagle"

import pandas
import math
import sys

from itertools import groupby


class ExpediaKNN:
    """
    Class for KNN algorithm that runs on N bootstrap samples and averages out the accuracy. (Bagging on KNN)
    """

    K_VALUE = 20
    SAMPLES = 20

    train_data_vector = None
    train_sampled_vector = None

    test_data_vector = None
    test_sampled_vector = None

    result_vector = list()

    result_bag = list()

    train_csv = "data/train_10k_sample.csv"
    test_csv = "data/test_10k_sample.csv"

    def __init__(self):
        pass

    def load_data(self, train_csv=None, test_csv=None):
        """
        Function to load the train and test data.
        :param train_csv:
        :param test_csv:
        :return:
        """

        if train_csv is not None and test_csv is not None:
            self.train_data_vector = pandas.read_csv(train_csv)
            self.test_data_vector = pandas.read_csv(test_csv)

    def learn(self):
        """
        Learn function that does nothing fancy.
        :return:
        """
        self.load_data()

    def classify(self, label_index):
        """
        Classification function that classifies every test data point with distance measure w.r.t. training data points.
        :param label_index:
        :return:
        """

        for rand_state in xrange(1111, 1111 + self.SAMPLES):

            self.result_vector = list()

            self.train_sampled_vector = self.train_data_vector.sample(n=2000, random_state=rand_state)
            self.test_sampled_vector = self.test_data_vector.sample(n=200, random_state=(rand_state + 2000))

            self.train_sampled_vector = self.train_sampled_vector.reset_index(drop=True)
            self.test_sampled_vector = self.test_sampled_vector.reset_index(drop=True)

            # Stores distance vectors in format [class_label, distance]
            classification_vector = list()

            euclid_distance = 0
            euclid_sum_sq = 0
            for x in xrange(0, len(self.test_sampled_vector)):

                euclid_distance = 0.0
                class_vector = list()

                for y in xrange(0, len(self.train_sampled_vector)):
                    euclid_sum_sq = 0.0
                    for index in self.test_sampled_vector:
                        if index in self.train_sampled_vector:
                            if index != label_index:

                                test_data_value = self.test_sampled_vector.get(index)[x]
                                train_data_value = self.train_sampled_vector.get(index)[y]

                                sq_diff = math.pow(((test_data_value if not math.isnan(test_data_value) else 0.0)
                                                    - (train_data_value if not math.isnan(train_data_value) else 0.0)), 2)

                                # Sum all the squared differences between the feature values.
                                euclid_sum_sq += sq_diff
                        else:
                            raise "Test and Train do not match"
                            break

                    euclid_distance = math.sqrt(euclid_sum_sq)

                    vector = list()
                    vector.append(self.train_sampled_vector.get(label_index)[y])
                    vector.append(euclid_distance)

                    class_vector.append(vector)

                    print("RandNum Iteration: {0} | X = {1} | Y = {2} | {3}".format(rand_state, x, y, vector))

                # Getting sorted vector.
                class_vector.sort(key = lambda x : x[1])

                # Create list of labels.
                classification_vector = [label[0] for label in class_vector]

                # Get max label from top K_VALUE labels.
                class_label = max(groupby(sorted(classification_vector[0:self.K_VALUE])),
                                  key=lambda (x, v): (len(list(v)), - classification_vector.index(x)))[0]

                self.result_vector.append(class_label)

            self.result_bag.append(self.get_accuracy(label_index))

    def get_accuracy(self, label_index):
        """
        Returns the accuracy of the classification.
        :param label_index:
        :return:
        """

        match_count = 0.0

        print(self.result_vector)
        print (self.test_sampled_vector.get(label_index)[0:len(self.test_sampled_vector - 1)])

        for x in xrange(0, len(self.test_sampled_vector)):
            if int(self.result_vector[x]) == self.test_sampled_vector.get(label_index)[x]:
                match_count += 1.0

        return match_count/float(len(self.test_sampled_vector))

    def get_agg_accuracy(self):
        """
        Returns averaged accuracy.
        :return:
        """

        return sum(self.result_bag) / len(self.result_bag)


def main():

    if len(sys.argv) < 3:
        print("ERROR: Too few arguments provided!\nSyntax: expedia_knn_with_sampling.py <train_csv_path> <test_csv_path> [<K Value>] [<Number of Bootstraps>]")
        exit(1)

    knn = ExpediaKNN()

    knn.train_csv = sys.argv[1]
    knn.test_csv = sys.argv[2]

    if len(sys.argv) > 3:
        print("Setting K to " + sys.argv[3])
        knn.K_VALUE = int(sys.argv[3])

    if len(sys.argv) > 4:
        print("Setting Bootstrap Samples to " + sys.argv[4])
        knn.SAMPLES = int(sys.argv[4])


    knn.load_data(knn.train_csv, knn.test_csv)
    # knn.get_mean_vectors_from_train("hotel_cluster")
    knn.classify("hotel_cluster")
    print(knn.get_agg_accuracy())

if __name__ == "__main__":
    main()