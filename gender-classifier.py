#importing models
from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB

#train data [height,weight,shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#classifiers
clf = tree.DecisionTreeClassifier()
clf1 = svm.SVC()
clf2 = neighbors.KNeighborsClassifier()
clf3 = GaussianNB()

#training models
clf = clf.fit(X,Y)
clf1 = clf1.fit(X,Y)
clf2 = clf2.fit(X,Y)
clf3 = clf3.fit(X,Y)

#test data
X_test=[[184,84,44],[198,92,48],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
Y_test=['male','male','male','female','female','female','male','male']

#predictions 
prediction = clf.predict(X_test)
prediction1 = clf1.predict(X_test)
prediction2 = clf2.predict(X_test)
prediction3 = clf3.predict(X_test)

#accuracy
result = accuracy_score(Y_test, prediction)
result1 = accuracy_score(Y_test, prediction1)
result2 = accuracy_score(Y_test, prediction2)
result3 = accuracy_score(Y_test, prediction3)

#printing results
print("Decision Tree Classifier: ",prediction)
print("Accuracy: ",result)
print("SVM Classifier: ",prediction1)
print("Accuracy: ",result1)
print("KNeighbors Classifier: ",prediction2)
print("Accuracy: ",result2)
print("GaussianNB Classifier: ",prediction3)
print("Accuracy: ",result3)