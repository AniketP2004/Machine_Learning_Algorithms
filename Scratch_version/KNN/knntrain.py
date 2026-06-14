import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from main import KNN
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# knn = KNeighborsClassifier(n_neighbors=32)
# knn.fit(X_train, y_train)
# score = knn.score(X_test, y_test)
# print(score)
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k', s=20 )
# plt.show()


if __name__ == "__main__":
    
    clf = KNN(k=5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # print(predictions)
    acc = np.sum(predictions == y_test)/ len(y_test)
    print(acc)

