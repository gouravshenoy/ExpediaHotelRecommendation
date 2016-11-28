import math
import DecisionTreeCore.Constants
from collections import Counter

from pprint import pprint

class DataAnalyser:

    def analyseFeatures(self, dataSet, filterIndex=[], availableFeatures=[], featureDimension=2):
        """
        This function analyzes the specified data with filter and returns the feature with maximum information gain.
        It returns the feature with breakdown as follows-
        {Feature1 :{feature_Value1:(+Count, -Count), feature_value2:{(+count, -count)})}}
        :param dataSet: data for operation
        :param filterIndex: index of the feature to filter the data
        :param availableFeatures: list of available features to calculate information gain
        :param featureDimension: number of distinct values the class label has
        :return: feature break-down dictionary for feature with maximum information gain
        """
        # for index, data in enumerate(dataSet.getMatrix()):
        #     print '{index}, {list}'.format(index=index, list=data)

        # if no filterIndex, scan full dataSet,
        #   else, create filtered dataSet
        filtered_data = dataSet.getMatrix()
        if filterIndex:
            filtered_data = [dataSet.getMatrix()[i] for i in filterIndex]

        # this data-structure holds vital information
        #   about the features, incl pos,neg counts
        featureDict = {}

        for feature in availableFeatures:
            featureDict[feature] = {}
            for index, data in enumerate(filtered_data):
                label_index = len(data) - 1

                if data[feature] in featureDict[feature]:
                    featureDict[feature][data[feature]][data[label_index]] += 1
                else:
                    featureDict[feature][data[feature]] = [0] * featureDimension
                    featureDict[feature][data[feature]][data[label_index]] += 1

        featureDict = self.calculateInfoGain(featureDict=featureDict)
        # pprint(featureDict)

        # find the feature with max
        #   information gain
        max_gain = 0.0
        for key, value in featureDict.items():
            if value['info-gain'] >= max_gain:
                max_gain = value['info-gain']
                max_gain_feature = key

        return {max_gain_feature: featureDict[max_gain_feature]}

    def calculateInfoGain(self, featureDict):
        """
        Calculates information gain for specified feature.
        :param featureDict: feature dictionary
        :return: feature dictionary with added information gain
        """
        for featureIndex, dict in featureDict.items():
            # total_pos_count = 0
            # total_neg_count = 0

            # contains total count of each class labels in featureDict
            # initialized all = 0
            total_count_all_classes = [0] * 100

            for featureValue, featureCounts in dict.items():

                for label_index in range(0, 100):
                    total_count_all_classes[label_index] += featureCounts[label_index]

                # total_pos_count += featureCounts[True.__int__()]
                # total_neg_count += featureCounts[False.__int__()]

            # print total_pos_count, total_neg_count
            parent_entropy = self.calculateMultiClassEntropy(total_count_all_classes)

            # norm_prob = |Sv| / |S|
            norm_prob = float(sum(featureCounts)) / sum(total_count_all_classes)

            sum_feature_entropy = 0.0
            for featureValue, featureCounts in dict.items():
                # initialize to 0
                total_count_classes = [0] * 100
                for label_index in range(0, 100):
                    total_count_classes[label_index] += featureCounts[label_index]

                # we calculate sum of entropies
                #   of all feature values
                sum_feature_entropy += (norm_prob) * \
                                    self.calculateMultiClassEntropy(total_count_classes)

            # subtract parent entropy from
            #   sum of feature entropies to get info-gain
            dict['info-gain'] = parent_entropy - sum_feature_entropy
            featureDict[featureIndex] = dict

        # pprint(featureDict)
        return featureDict


    def calculateMultiClassEntropy(self, total_count_classes):
        """
        Calculates entropy from multi-class counts specified.
        :param total_count_classes:
        :return:
        """

        # list to hold probabilities of class labels
        probability_list = [0] * len(total_count_classes)

        # check if atleast one count == 0,
        #   then entropy is 0
        entropy = 0
        if all(v > 0 for v in total_count_classes):
            for index, value in enumerate(total_count_classes):
                probability_list[index] = float(total_count_classes[index]) \
                                          / sum([x for x in total_count_classes if x != total_count_classes[index]])

            for probability in probability_list:
                entropy += -(probability * math.log(probability, 2))

        return entropy

    def calculateEntropy(self, posCount=0, negCount=0):
        """
        Calculates entropy from positive and negative counts specified.
        :param posCount: positive class label count
        :param negCount: negative class label count
        :return: entropy for the data
        """
        # if either count is 0,
        #   then entropy is 0
        if posCount>0 and negCount>0:
            p_pos = float(posCount) / (posCount + negCount)
            p_neg = float(negCount) / (posCount + negCount)
            entropy = -(p_pos * math.log(p_pos, 2)) - (p_neg * math.log(p_neg, 2))
        else:
            entropy = 0
        return entropy
