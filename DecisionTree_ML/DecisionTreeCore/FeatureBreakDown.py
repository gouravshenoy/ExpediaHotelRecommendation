class FeatureBreakDown:

    def __init__(self):
        self.featureValues= []
        self.positiveCount = {}
        self.negativeCount = {}
        self.postiveWeights = {}
        self.negativeWeights = {}
        self.totalErrorWeight = 0
        self.errorWeights = {}
        self.predictedLabel = {}

    def initialize_feature_value(self, feature_value):
        self.featureValues.append(feature_value)
        self.positiveCount[feature_value] = 0
        self.negativeCount[feature_value] = 0
        self.postiveWeights[feature_value] = 0
        self.negativeWeights[feature_value] = 0
        self.errorWeights[feature_value] = 0
        self.predictedLabel[feature_value] = 0