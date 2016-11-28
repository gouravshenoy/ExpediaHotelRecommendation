import math
from DecisionTreeCore import Constants
from collections import Counter
from DecisionTreeCore.FeatureBreakDown import FeatureBreakDown
import copy
from pprint import pprint

class WeightedDataAnalyser:

    def analyseFeatures(self, dataSet, weights, filterIndex=[], availableFeatures=[]):
        """
        This function analyzes the specified data with filter and returns the feature with maximum information gain.
        It returns the feature with breakdown as follows-
        {Feature1 :{feature_Value1:(+Count, -Count), feature_value2:{(+count, -count)})}}
        :param dataSet: data for operation
        :param filterIndex: index of the feature to filter the data
        :param availableFeatures: list of available features to calculate information gain
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
            featureBreakDown = FeatureBreakDown()
            for index, data in enumerate(filtered_data):
                label_index = len(data) - 1
                featureValue = data[feature]
                label = data[label_index]
                if featureValue not in featureBreakDown.featureValues:
                    featureBreakDown.initialize_feature_value(featureValue)

                if label == 0:
                    featureBreakDown.negativeCount[featureValue] += 1
                    featureBreakDown.negativeWeights[featureValue] += weights[index]
                else:
                    featureBreakDown.positiveCount[featureValue] += 1
                    featureBreakDown.postiveWeights[featureValue] += weights[index]

            for featureValue in featureBreakDown.featureValues:
                if featureBreakDown.positiveCount[featureValue] < featureBreakDown.negativeCount[featureValue]:
                    featureBreakDown.predictedLabel[featureValue] = 0
                    featureBreakDown.errorWeights[featureValue] = featureBreakDown.postiveWeights[featureValue]
                    featureBreakDown.totalErrorWeight += featureBreakDown.postiveWeights[featureValue]
                else:
                    featureBreakDown.predictedLabel[featureValue] = 1
                    featureBreakDown.errorWeights[featureValue] = featureBreakDown.negativeWeights[featureValue]
                    featureBreakDown.totalErrorWeight += featureBreakDown.negativeWeights[featureValue]

            featureDict[feature] = featureBreakDown


        minErrorWeight = 999999
        minErrorWeightFeature = -1
        for feature in availableFeatures:
            featureBreakDown = featureDict[feature]
            if featureBreakDown.totalErrorWeight < minErrorWeight:
                minErrorWeight = featureBreakDown.totalErrorWeight
                minErrorWeightFeature = feature

        return minErrorWeightFeature, featureDict

'''
        for feature in availableFeatures:
            featureDict[feature] = {}
            for index, data in enumerate(filtered_data):
                label_index = len(data) - 1

                if data[feature] in featureDict[feature]:
                    featureDict[feature][data[feature]][data[label_index]] += 1
                else:
                    featureDict[feature][data[feature]] = [0,0]
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
            total_pos_count = 0
            total_neg_count = 0
            for featureValue, featureCounts in dict.items():
                total_pos_count += featureCounts[True.__int__()]
                total_neg_count += featureCounts[False.__int__()]

            # print total_pos_count, total_neg_count
            parent_entropy = self.calculateEntropy(total_pos_count, total_neg_count)

            sum_feature_entropy = 0.0
            for featureValue, featureCounts in dict.items():
                # norm_prob = |Sv| / |S|
                norm_prob = float(sum(featureCounts)) / (total_pos_count + total_neg_count)

                # we calculate sum of entropies
                #   of all feature values
                sum_feature_entropy += (norm_prob) * \
                                    self.calculateEntropy(featureCounts[True.__int__()],
                                                          featureCounts[False.__int__()])

            # subtract parent entropy from
            #   sum of feature entropies to get info-gain
            dict['info-gain'] = parent_entropy - sum_feature_entropy
            featureDict[featureIndex] = dict

        # pprint(featureDict)
        return featureDict


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
        return entropy'''
