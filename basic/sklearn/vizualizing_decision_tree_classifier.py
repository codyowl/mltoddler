import sklearn
from sklearn import tree
#importing irist dataset
from sklearn.datasets import load_iris
import numpy as np
from sklearn.

iris = load_iris()

#Categorising the dataset for training and testing
test_idx = [0,50,100]

#training data
#we are going to remove three values from the dataset
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
#we are going to use the rest of the things removed from the training dataset as testing dataset
test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

classifier = tree.DecisionTreeClassifier()

#training on our training data
classifier.fit(train_data, train_target)

print "target data"
print test_target

#calling the prediction method and passing test_data which should the return the appropriate test target value
print classifier.predict(test_data)

#Now for vizualising our decision tree we have to call the "export_graphviz" method


