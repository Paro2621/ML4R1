# import knn classifier model
from sklearn.neighbors import KNeighborsClassifier
# import data loader
from sklearn.datasets import load_wine
# import performance metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import utility function to split the dataset
from sklearn.model_selection import train_test_split


# Load the Wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Predicting the test set results
y_pred = knn_model.predict(X_test)

# Evaluating the model
accuracy_knn = accuracy_score(y_test, y_pred)
precision_knn = precision_score(y_test, y_pred, average='macro')
recall_knn = recall_score(y_test, y_pred, average='macro')
f1_knn = f1_score(y_test, y_pred, average='macro')

# Print the results
print("Accuracy:", accuracy_knn)
print("Precision:", precision_knn)
print("Recall:", recall_knn)
print("F1 Score:", f1_knn)




# plot
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
pca = PCA(n_components=2)  # Reducing to 2 dimensions for visualisation
pca.fit(X_train)
X_pca = pca.transform(X_test)
X_error = X_pca[y_test != y_pred,:]

colors = ['red','green','blue']

plt.scatter(X_pca[:,0], X_pca[:,1], s=12, marker='x', c=y_test, cmap=ListedColormap(colors))
plt.plot(X_error[:,0], X_error[:,1], 'ok', markersize=15, fillstyle='none')
plt.grid()
plt.show()
