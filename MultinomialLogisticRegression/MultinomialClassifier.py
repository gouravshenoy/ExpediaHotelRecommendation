import numpy as np
from scipy.optimize import fmin_cg
import pandas as pd
import pickle

class Storage:

	def __init__(self, name, model, predicted_labels, mask, accuracy, probabilites):
		self.name = name
		self.model = model
		self.predicted_labels =  predicted_labels
		self.mask = mask
		self.accuracy = accuracy
		self.probabilities = probabilites

"""
This is class for Multinomial logistic regression which holds the related methods and data for performing multinomial Logistic Regression. It has fit(..) and predict(..) methods which is standard signature methods for machine learning classifiers. 
"""
class MultinomialLogisticRegression:
    
    def __init__(self):
        self.X = None
        self.Y = None
        self.no_of_features = None
        self.no_of_examples = None
        self.no_of_classes = None
        self.learned_weights = None
        
        self.cost = None
        self.gradient = None
        self.predicted_probabilities = []
    
    def fit(self, X, y):
        # Check the number of training examples are equal to labels provided
        if(len(X) != len(y)):
            print('Label and example count mis-match')
            return None
        
        self.no_of_examples = X.shape[0] 
        self.no_of_features = X.shape[1]
        self.classes = np.unique(y)
        self.no_of_classes = len(self.classes)
        
        # Binarize the classes convert classes into columns with each example having value 0 or 1 
        Y = np.zeros((self.no_of_examples, self.no_of_classes), dtype=np.float64)
        for i, class_value in enumerate(self.classes):
            Y[y == class_value, i] = 1

        self.X = X
        self.Y = Y
        
        # Initialize the weights with random values 
        np.random.seed(1234)
        w0 = np.random.random((self.no_of_features * self.no_of_classes, ))


        # Find the optimal weights with Conjugate Gradient from SciPy
        results = fmin_cg(self.cost_function, w0, fprime=self.gradient_function, full_output = True)
        self.learned_weights = results[0].reshape((self.no_of_features, self.no_of_classes))

        # Return self as the trained model
        return self
    
    def predict(self, X):
        
        if(X.shape[1] !=  self.no_of_features):
            print("Mismatch between features in Training data and prediction data")
        
        Yhat = np.dot(X, self.learned_weights)
        predicted_probabilities = Yhat / Yhat.sum(axis=1)[:, np.newaxis]
        self.predicted_probabilities.append(predicted_probabilities)
        Yhat -= Yhat.min(axis=1)[:, np.newaxis]
        Yhat = np.exp(-Yhat)
        Yhat /= Yhat.sum(axis=1)[:, np.newaxis]
        yhat = self.classes[np.argmax(Yhat, axis=1)]
        return yhat
    
    def calculate_functions(self, w):

        W = w.reshape(self.no_of_features, self.no_of_classes)
        
        X = self.X
        Y = self.Y
        
        Yhat = np.dot(X, W)
        Yhat -= Yhat.min(axis=1)[:, np.newaxis]
        Yhat = np.exp(-Yhat)
        Yhat /= Yhat.sum(axis=1)[:, np.newaxis]

        _Yhat = Yhat * Y
        cost = np.sum(np.log(_Yhat.sum(axis=1)))
        _Yhat /= _Yhat.sum(axis=1)[:, np.newaxis]
        Yhat -= _Yhat

        gradient = np.dot(X.T, Yhat)

        cost /= -float(self.no_of_examples)
        gradient /= -float(self.no_of_examples)

        self.cost = cost
        self.gradient = gradient

    def cost_function(self, w):
        self.calculate_functions(w)
        cost = self.cost
        self.cost = None
        return cost

    def gradient_function(self, w):
        self.calculate_functions(w)
        gradient = self.gradient.ravel()
        self.gradient = None  
        return gradient


"""
 NOTE: This line onwards is the main program which takes default file path as input and executes the data
"""
import sys
def main():
	print('Starting Program...')

	# Please update file path accordingly this file should contain all the 
	#file_path = '/home/anwar/AML_Project/data/train_booking_dest_merged_cluster_100.csv'
	file_path = sys.argv[1]
	expedia_data = pd.read_csv(file_path)
	expedia_data = expedia_data.fillna(0)

	# Uncomment this line if you passing unfiltered data with following non-numeric columns
	#expedia_data = expedia_data.drop(labels=['date_time', 'srch_ci', 'srch_co'], axis=1)
	hotel_clusters = ['hotel_cluster5', 'hotel_cluster10']
	expedia_X = expedia_data.drop(labels=hotel_clusters, axis=1)

	for hotel_cluster in hotel_clusters:

		expedia_y = expedia_data[hotel_cluster]

		msk = np.random.rand(len(expedia_data)) < 0.8

		expedia_X_train = expedia_X[msk]
		expedia_X_test = expedia_X[~msk]

		expedia_y_train = expedia_y[msk]
		expedia_y_test = expedia_y[~msk]


		multinomialLogisticRegression = MultinomialLogisticRegression()
		multinomialLogisticRegression.fit(expedia_X_train, expedia_y_train)

		y_hat = multinomialLogisticRegression.predict(expedia_X_test)

		# Uncomment for creating a pickle object on the disk for the classifier.
		#storage = Storage(name, multinomialLogisticRegression, y_hat, 	msk, accuracy, multinomialLogisticRegression.predicted_probabilities)

	# Using accuracy score metric from scikit learn for accuracy calculations
	from sklearn.metrics import accuracy_score
	print(accuracy_score(expedia_y_test, y_hat))
	print(multinomialLogisticRegression.predicted_probabilities)

	#pickle.dump(storage, open( "./../models/" + hotel_cluster, "wb" ) )

if __name__ == "__main__":
    main()


