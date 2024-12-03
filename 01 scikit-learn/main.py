from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Cargar los datos de iris
iris = load_iris()

#Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

#Crear un clasificador KNN con k=3
knn = KNeighborsClassifier(n_neighbors=3)

#Entrenar el clasificador        
knn.fit(X_train, y_train)

#Calcular la precisión del clasificador en los datos de prueba
print("Precisión del clasificador en los datos de prueba de iris:{:.2f}".format(knn.score(X_test, y_test)))