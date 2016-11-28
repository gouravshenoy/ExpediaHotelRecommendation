# imports used for drawing DT (Optional)
import uuid
import pydot
import random

# imports for computation
from DecisionTreeCore.DataAnalyser import DataAnalyser
from DecisionTreeCore import Constants


class DecisionTree:
    """
    This is class responsible for creating and maintaining decision tree
    """
    __rootNode = -1
    __treeDepth = -1

    def train(self, data, treeDepth):
        """
        This function train the decision tree model with specified depth on the given data
        :param data: training data
        :param treeDepth: depth of the tree
        :return: none
        """
        indices = []
        availableFeatures = data.getAvailableFeatures()

        self.__treeDepth = treeDepth
        self.__rootNode = self.createNode(data, indices, availableFeatures, 0)

    def createNode(self, data, subsetIndices, availableFeatures, nodeDepth, positiveCount=-1, negativeCount=-1):
        """
        Creates a node in the tree and returns the node object
        :param data: data object
        :param subsetIndices: indices of data point at the node level for operation
        :param availableFeatures: list available features at the node for split
        :param nodeDepth: depth of the node
        :param positiveCount: total positive values in the data
        :param negativeCount: total negative values in the data
        :return:
        """
        dataAnalyser = DataAnalyser()
        # print("\nNode Data--")
        # print("Subset Indices: ", subsetIndices)
        # print("Available Features: ", availableFeatures)

        # Check if the node has pure class
        if (positiveCount == 0 or negativeCount == 0):
            # Creating a pure class leaf node and return
            positiveRatio = positiveCount / (positiveCount + negativeCount)
            negativeRatio = negativeCount / (negativeCount + positiveCount)
            node = Node(-1, positiveRatio, negativeRatio, nodeDepth)
            return node

        # If node is not pure get the feature breakdown from analyzer
        featureBreakDown = dataAnalyser.analyseFeatures(data, subsetIndices, availableFeatures, Constants.FEATURE_DIMENSION)
        # print("Feature Breakdown from Analyzer: ")
        # pprint(featureBreakDown)

        feature = list(featureBreakDown.keys())[0]
        featureValues = list(featureBreakDown.get(feature).keys())
        featureValues.remove('info-gain')

        # calculate positive and negative class ratio at the node
        if (positiveCount == -1 or negativeCount == -1):
            negativeCount = 0
            positiveCount = 0
            for featureValue in featureValues:
                negativeCount += featureBreakDown.get(feature).get(featureValue)[0]
                positiveCount += featureBreakDown.get(feature).get(featureValue)[1]

        positiveRatio = float(positiveCount) / (positiveCount + negativeCount)
        negativeRatio = float(negativeCount) / (negativeCount + positiveCount)

        # Create a new node
        node = Node(feature, positiveRatio, negativeRatio, nodeDepth)

        # Create branches and child nodes for each possible value feature can take
        childrenAvailableFeatures = list(availableFeatures)
        childrenAvailableFeatures.remove(feature)
        childrenNodeDepth = nodeDepth + 1
        if self.isTerminationCondition(childrenNodeDepth, positiveRatio, childrenAvailableFeatures) == False:
            for featureValue in featureValues:
                childSubsetIndices = data.getDataIndices(feature, featureValue, subsetIndices)
                childNode = self.createNode(data, childSubsetIndices,
                                            childrenAvailableFeatures, childrenNodeDepth,
                                            featureBreakDown.get(feature).get(featureValue)[1],
                                            featureBreakDown.get(feature).get(featureValue)[0])
                node.addChildren(featureValue, childNode)

        return node

    def isTerminationCondition(self, childNodeDepth, positiveRatio, childrenAvailableFeatures):
        """
        Check if the termination condition is reached in tree expansion process which is one of following-
        1. Pure class is created
        2. Maximum depth node is created
        3. No features are available to split
        :param childNodeDepth: would be depth of the child node
        :param positiveRatio: positive class ratio at the node
        :param childrenAvailableFeatures: features available for child node to split
        :return: True if termination condition reached or else False.
        """
        isTerminationCondition = False
        if (positiveRatio == 1
            or positiveRatio == 0
            or childNodeDepth > self.__treeDepth
            or not childrenAvailableFeatures):
            isTerminationCondition = True

        # print('Is Terminated: ', isTerminationCondition)
        return isTerminationCondition

    def test(self, data):
        """
        function test the provided data with created model and returns predicted class label vector.
        :param data: data to test
        :return: list of predicted class labels for the data
        """
        rootNode = self.__rootNode
        predictedLabel = []

        for dataPoint in data:
            currentNode = rootNode
            while (currentNode.getChildren() != -1):
                featureToTest = currentNode.getFeatureIndex()
                dataPointValue = dataPoint[featureToTest]
                children = currentNode.getChildren()
                if dataPointValue in children:
                    currentNode = children[dataPointValue]
                else:
                    break

            predictedLabel.append(currentNode.getClassLabel())
        return predictedLabel

    def calculateAccuracy(self, dataSet, predictedLabels):
        """
        function calculates accuracy and mis-classification count for performed experiment
        :param dataSet: test data-set with actual labels
        :param predictedLabels: predicted labels on the data
        :return: %accuracy and mis-classification count
        """
        correctClassCount = 0
        dataSetLen = len(dataSet)

        for index, dataPoint in enumerate(dataSet):
            if dataPoint[len(dataPoint) - 1] == predictedLabels[index]:
                correctClassCount += 1

        accuracy = float(correctClassCount) / dataSetLen * 100
        misclassification = dataSetLen - correctClassCount
        return accuracy, misclassification

    def plotConfusionMatrix(self, dataSet, predictedLabels):
        """
        This function draws the confusion matrix for the data
        :param dataSet: test data-set with the actual labels
        :param predictedLabels: list of predicted labels for the data
        :return: none
        """
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        for index, dataPoint in enumerate(dataSet):
            if dataPoint[len(dataPoint) - 1] == predictedLabels[index]:
                if (predictedLabels[index] == 1):
                    true_pos += 1
                else:
                    true_neg += 1
            else:
                if (predictedLabels[index] == 1):
                    false_pos += 1
                else:
                    false_neg += 1

        print ('\n- Confusion Matrix:')
        print ('{0:{align}{width}} '
               '{1:{align}{width2}}'.format(' ',
                                            ' predicted label ',
                                            width=10,
                                            width2=50,
                                            align='^'))
        print ('{0:{fill}{align}{width}} '.format('-',
                                                  fill='-',
                                                  width=65,
                                                  align='^'))
        print ('{0:{align}{width2}}'
               '| {1:{align}{width}} | '
               '{2:{align}{width}} | '.format('',
                                              'label = 0 ',
                                              'label = 1',
                                              width2=24,
                                              width=10,
                                              align='^'))
        print ('{0:{fill}{align}{width}} '.format('-',
                                                  fill='-',
                                                  width=65,
                                                  align='^'))
        print ('{0:{align}{width}} | '
               '{1:{align}{width}} | '
               '{2:{align}{width}} | '
               '{3:{align}{width}} | '
               '{4:{align}{width}} | '.format(' ',
                                              'label = 0',
                                              'TN: ' + str(true_neg),
                                              'FP: ' + str(false_pos),
                                              'Total = ' + str(true_neg + false_pos),
                                              width=10,
                                              align='^'))
        print ('{0:{align}{width}} '
               '{1:{fill}{align}{width2}}'.format('true label',
                                                  '-',
                                                  fill='-',
                                                  width=10,
                                                  width2=54,
                                                  align='^'))
        print ('{0:{align}{width}} | '
               '{1:{align}{width}} | '
               '{2:{align}{width}} | '
               '{3:{align}{width}} | '
               '{4:{align}{width}} | '.format(' ',
                                              'label = 1',
                                              'FN: ' + str(false_neg),
                                              'TP: ' + str(true_pos),
                                              'Total = ' + str(false_neg + true_pos),
                                              width=10,
                                              align='^'))
        print ('{0:{fill}{align}{width}} '.format('-',
                                                  fill='-',
                                                  width=65,
                                                  align='^'))
        print ('{0:{align}{width2}} '
               '{1:{align}{width}} | '
               '{2:{align}{width}} | '
               '{3:{align}{width}} | '.format('',
                                              'Total = ' + str(true_neg + false_neg),
                                              'Total = ' + str(false_pos + true_pos),
                                              str(true_neg + false_neg + false_pos + true_pos),
                                              width2=24,
                                              width=10,
                                              align='^'))
        print ('{0:{fill}{align}{width}} '.format('-',
                                                  fill='-',
                                                  width=65,
                                                  align='^'))

    def printTree(self, op_file=None):
        """
        This function creates the tree image and writes it on disk
        :return: none
        """
        print('Saving decision tree visualization in dt_plots folder!')
        self.__graph = pydot.Dot(graph_type='graph')
        self.__rootNode.drawNode(graph=self.__graph)
        self.__graph.write_png('/Users/goshenoy/dt-plots/' + op_file + '-depth-' + str(self.__treeDepth) + '.png')


class Node:
    """
    This class is the Node in the tree. Holds the required values and getter, setter  and modification
    functions for the node.
    """
    __featureIndex = -1
    __children = {}
    __positiveRatio = 0
    __negativeRatio = 0
    __nodeDepth = -1

    def __init__(self, feature, positiveRatio, negativeRatio, nodeDepth):
        self.__featureIndex = feature
        self.__positiveRatio = positiveRatio
        self.__negativeRatio = negativeRatio
        self.__nodeDepth = nodeDepth
        self.__children = -1

    def addChildren(self, featureValue, childNode):
        if self.__children == -1:
            self.__children = {}

        self.__children[featureValue] = childNode

    def getPositiveRatio(self):
        return self.__positiveRatio

    def negativeRatio(self):
        return self.__negativeRatio

    def getNodeDepth(self):
        return self.__nodeDepth

    def getChildren(self):
        return self.__children

    def getFeatureIndex(self):
        return self.__featureIndex

    def getClassLabel(self):
        if self.__positiveRatio > self.__negativeRatio:
            return 1
        elif self.__positiveRatio < self.__negativeRatio:
            return 0
        else:
            return random.choice((1,0))

    def printNode(self):
        print("\n\nNode Feature:", self.getFeatureIndex(), "| ClassLabel:", self.getClassLabel())
        if self.getChildren() != -1:
            print("Node Children: ")
            for branchValue in self.getChildren():
                # print("Node-",node)
                print("Child:: Branch-value:", branchValue,
                      " | Feature:", self.getChildren()[branchValue].getFeatureIndex())

            for node in self.getChildren().values():
                node.printNode()
        else:
            print("NO CHILD")

    def drawNode(self, graph):
        node_name = "Feature: " + str(self.getFeatureIndex()) \
                    + "\nClass-Label: %d" % self.getClassLabel() \
                    + "\nDepth: %d" % self.getNodeDepth() \
                    + "\nPos-Ratio: " + str(self.getPositiveRatio()) \
                    + "\nNeg-Ratio: " + str(self.negativeRatio())

        if self.getChildren() != -1:
            dt_node = pydot.Node(str(uuid.uuid4()),
                                 style="filled",
                                 fillcolor="yellow",
                                 label=node_name)

            for branchValue in self.getChildren():
                child_node = self.getChildren()[branchValue].drawNode(graph=graph)
                graph.add_edge(pydot.Edge(dt_node, child_node, label=str(branchValue)))
        else:
            dt_node = pydot.Node(str(uuid.uuid4()),
                                 style="filled",
                                 fillcolor="green",
                                 label=node_name)

        graph.add_node(dt_node)
        return dt_node
