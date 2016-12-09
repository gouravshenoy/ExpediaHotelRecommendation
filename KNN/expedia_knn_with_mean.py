__author__ = "Mangirish Wagle"

import pandas
import math

from itertools import groupby


class ExpediaKNN:
    """
    Class for KNN algorithm.
    """

    K_VALUE = 20

    train_data_vector = None
    mean_data_vector = None
    test_data_vector = None
    result_vector = list()

    train_csv = "data/train_10k_sample_trunc.csv"
    test_csv = "data/test_10k_sample_trunc.csv"

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

            # print(sorted(self.train_data_vector["hotel_cluster"].unique()))

            # print(self.train_data_vector[self.train_data_vector["hotel_cluster"] == 2]["user_location_country"].mean())

            # print(list(self.train_data_vector.columns))

    def get_mean_vectors_from_train(self, label_index):

        append_list = list(list(self.train_data_vector.columns))
        append_list.remove(label_index)

        self.mean_data_vector = pandas.DataFrame(columns=list(self.train_data_vector.columns))
        for label in sorted(self.train_data_vector[label_index].unique()):
            self.mean_data_vector.loc[len(self.mean_data_vector)]\
                = [(self.train_data_vector[self.train_data_vector[label_index]
                                           == label][col].mean()) for col in list(self.train_data_vector.columns)]

        # print([int(x) for x in self.mean_data_vector[label_index]])

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

        # Stores distance vectors in format [class_label, distance]
        classification_vector = list()

        euclid_distance = 0
        euclid_sum_sq = 0
        for x in xrange(0, len(self.test_data_vector)):

            euclid_distance = 0.0
            class_vector = list()

            for y in xrange(0, len(self.mean_data_vector)):
                euclid_sum_sq = 0.0
                for index in self.test_data_vector:
                    if index in self.mean_data_vector:
                        if index != label_index:

                            test_data_value = self.test_data_vector.get(index)[x]
                            mean_data_value = self.mean_data_vector.get(index)[y]

                            sq_diff = math.pow(((test_data_value if not math.isnan(test_data_value) else 0.0)
                                                - (mean_data_value if not math.isnan(mean_data_value) else 0.0)), 2)

                            # Sum all the squared differences between the feature values.
                            euclid_sum_sq += sq_diff
                    else:
                        raise "Test and Train do not match"
                        break

                euclid_distance = math.sqrt(euclid_sum_sq)

                vector = list()
                vector.append(self.mean_data_vector.get(label_index)[y])
                vector.append(euclid_distance)

                class_vector.append(vector)

                print("X = {0} | Y = {1} | {2}".format(x, y, vector))

            # Getting sorted vector.
            class_vector.sort(key = lambda x : x[1])

            # Create list of labels.
            classification_vector = [label[0] for label in class_vector]

            # Get max label from top K_VALUE labels.
            class_label = max(groupby(sorted(classification_vector[0:self.K_VALUE])),
                              key=lambda (x, v): (len(list(v)), - classification_vector.index(x)))[0]
    
            self.result_vector.append(class_label)

    def get_accuracy(self, label_index):
        """
        Returns the accuracy of the classification.
        :param label_index:
        :return:
        """

        match_count = 0.0

        print(self.result_vector)
        print (self.test_data_vector.get(label_index)[0:len(self.test_data_vector - 1)])

        for x in xrange(0, len(self.test_data_vector)):
            if int(self.result_vector[x]) == self.test_data_vector.get(label_index)[x]:
                match_count += 1.0

        return match_count/float(len(self.test_data_vector))


def main():
    knn = ExpediaKNN()

    knn.load_data(knn.train_csv, knn.test_csv)
    knn.get_mean_vectors_from_train("hotel_cluster")
    knn.classify("hotel_cluster")
    print(knn.get_accuracy("hotel_cluster"))

if __name__ == "__main__":
    main()