# Importing the required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Loading the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the classification model (using k-nearest neighbors in this example)
knn_model = KNeighborsClassifier(n_neighbors=3)

# Training the model
knn_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = knn_model.predict(X_test)

# Evaluating model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Viewing the ranking report
print('\nRelatório de Classificação:')
print(classification_report(y_test, y_pred))