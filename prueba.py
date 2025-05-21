# %%
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
from skimage.feature import hog
import re

# %%
# Directorio de imágenes
image_dir = "billetes/"
labels = []
features = []

# %%
for file in os.listdir(image_dir):
    if file.endswith(".jpg") or file.endswith(".JPG"):
        # Usamos una expresión regular para extraer el número después de 'MXN' y antes de 'N'
        match = re.search(r'MX(\d+)N', file)
        if match:
            label = int(match.group(1))  # Extraemos el número como entero
            print(f"Etiqueta extraída: {label}")  # Imprimir para verificar
        else:
            print(f"No se pudo extraer la etiqueta de: {file}")  # Si no se extrae la etiqueta, lo imprimimos
            continue
        
        img_path = os.path.join(image_dir, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))  # Redimensionar
        # Extraer características HOG
        hog_features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True)
        features.append(hog_features)
        labels.append(label)



# %%
# Convertir a arreglos numpy
X = np.array(features)
Y = np.array(labels)

# %%
# Dividir en entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# %%
# Entrenar un modelo SVM
modelo = SVC(kernel='linear')
modelo.fit(X_train, Y_train)

# %%
# Guardar el modelo entrenado
with open("modelo_billetes.pkl", "wb") as archivo_modelo:
    pickle.dump(modelo, archivo_modelo)

# %%
# Cargar el modelo previamente guardado
with open("modelo_billetes.pkl", "rb") as archivo_modelo:
    modelo_cargado = pickle.load(archivo_modelo)

# %%
# Evaluar el modelo
y_pred = modelo_cargado.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

# Matriz de confusión
cm = confusion_matrix(Y_test, y_pred)
print("Matriz de Confusión:")
print(cm)

# Reporte de clasificación
print("Reporte de clasificación:")
print(classification_report(Y_test, y_pred))


