import pandas as pd
from pprint import pprint


class NaiveBayes:

    def __init__(self, train_file, test_file):
        self.__probabilities = {}
        self.__train_file = train_file
        self.__test_file = test_file
        self.read_file()

    def read_file(self):
        # read data into data-frame
        print ('Reading data into pandas dataframe')
        self.__train_data = pd.read_csv(self.__train_file)
        self.__test_data = pd.read_csv(self.__test_file)

        # get counts & probs of labels (100 labels)
        self.__labels_count = {}
        self.__labels_probability = {}

        print ('Calculating class label counts')
        label_aggregation = self.__train_data.groupby(self.__train_data.hotel_cluster).size()
        for label, label_count in label_aggregation.iteritems():
            self.__labels_count[label] = label_count
            self.__labels_probability[label] = float(label_count + 1) / float(len(self.__train_data) + 100)    # laplace

        return


    def calculate_probabilities(self):
        for feature in list(self.__train_data.columns.values):
            self.__probabilities[feature] = {}
            print ('Calculating probability for feature: {}'.format(feature))

            # iterate over all values for that feature
            for feature_value in self.__train_data[feature].unique():
                self.__probabilities[feature][feature_value] = {}

                # iterate over all class labels
                for class_label in self.__labels_count:
                    # count (feature=feature_value & class=class_value)
                    feature_count = self.__train_data[
                        (self.__train_data[feature] == feature_value) &
                        (self.__train_data.hotel_cluster == class_label)] \
                    .groupby(feature).size()

                    if not (len(feature_count) == 1):
                        # print('feature: {}, value: {}, cluster: {}'.format(feature, feature_value, class_label))
                        feature_count = 0
                    else:
                        feature_count = feature_count.iloc[0]

                    # calculate probability (laplace correction)
                    probability = float(feature_count + 1) / \
                                    float(self.__labels_count[class_label] + len(self.__labels_count))
                    self.__probabilities[feature][feature_value][class_label] = probability
        return


    def predict(self):
        columns = self.__test_data.columns.values
        self.__predicted_labels = {}

        # iterate through every row
        for index, row in self.__test_data.iterrows():
            max_prob = 0

            for class_label in self.__labels_probability:
                prob_product = 1

                for feature in columns:
                    feature_value = row[feature]
                    if (feature_value in self.__probabilities[feature]):
                        prob_product *= self.__probabilities[feature][feature_value][class_label]
                    else:
                        prob_product = 0

                # check if max prob, if so add to predicted_labels
                if prob_product > max_prob:
                    max_prob = prob_product
                    self.__predicted_labels[index] = class_label
        return


    def get(self):
        return self.__predicted_labels

if __name__ == '__main__':
    train_file = '../../../Datasets/cleaned_and_merged/train_8k_cleaned.csv'
    test_file = '../../../Datasets/cleaned_and_merged/test_2k_cleaned.csv'

    nb = NaiveBayes(train_file, test_file)
    nb.calculate_probabilities()
    nb.predict()
    pprint(nb.get())