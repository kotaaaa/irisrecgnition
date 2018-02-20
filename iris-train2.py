import pandas as pd
from sklearn import svm, metrics, cross_validation

csv = pd.read_csv('iris.csv')

# csv_data = csv[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
# csv_label = csv["variety"]

# csv_data = csv[[5.1], ["sepal.width", "petal.length", "petal.width"]]
csv_data = csv.ix[ :,0:4]
csv_label = csv["variety"]

print(csv_data)
print(csv_label)

train_data, test_data, train_label, test_label = cross_validation.train_test_split(csv_data, csv_label)

clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)


ac_score = metrics.accuracy_score(test_label, pre)
print("正解率:", ac_score)
