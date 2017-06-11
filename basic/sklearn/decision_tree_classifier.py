import sklearn

#Consider a table where we have the following things
#
"""
Weight Texture Label
150g   Bumpy   Orange
170g   Bumpy   Orange
140g   Smooth  Apple
130g   Smooth  Apple
"""

#we are going to use two variable 'feautres' and 'labels'
#Feature contains the first two column(weight, texture) and labels contains the last column
features = [
    #0 is bumpy, and 1 is smooth
    [140, 1],
    [130, 1],
    [170, 0],
    [150, 0]
]

labels = [0, 0, 1, 1] #0 is apple and 1 is orange

#Features is the input to the classifier and labels is the output to the class

#Training our classifier
#we are going to use Decision tree classifier
from sklearn import tree

classifier = tree.DecisionTreeClassifier()

#fit is the training algorithm which is include in the classifier object
classifier.fit(features, labels)

#lets call the predict method to predict the output
print classifier.predict([[160, 0]])
