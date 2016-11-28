
class Data:
    """
    This class holds the data matrix and related function
    """
    def __init__(self):
        self.__matrix = []

    def getMatrix(self):
        return self.__matrix

    def setMatrix(self, matrix):
        self.__matrix = matrix

    def getAvailableFeatures(self):
        """
        This function returns the available features in the data.
        :return: list of available features
        """
        matrix = self.__matrix
        return list(range(0,len(matrix[0])-1))

    def getDataIndices(self, featureIndex, featureValue, subsetIndices):
        """
        This function list of indices in the data (subset of data) having particular feature value.
        :param featureIndex: feature index to subset
        :param featureValue: value of the feature to compare
        :param subsetIndices: list of current indices (current subset of the data)
        :return: list of indices
        """
        indices = []

        if len(subsetIndices) == 0:
            subsetIndices = list(range(0, len(self.__matrix)))

        for dataPointIndex in subsetIndices:
            dataPoint = self.__matrix[dataPointIndex]
            if dataPoint[featureIndex] == featureValue:
                indices.append(dataPointIndex)

        return indices
