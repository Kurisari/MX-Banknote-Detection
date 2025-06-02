import cv2
import numpy as np
import pickle
import os
from skimage.feature import hog

# Ruta del modelo
ruta_modelo = os.path.join(os.path.dirname(__file__), "modelo_completo.pkl")

# Cargar el modelo entrenado
with open(ruta_modelo, "rb") as archivo_modelo:
    datos = pickle.load(archivo_modelo)
    modelo = datos["modelo"]
    scaler = datos["scaler"]

def extraer_caracteristicas(img):
    img = cv2.resize(img, (128, 128))
    hog_features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True)
    return np.array(hog_features).reshape(1, -1)

def obtener_componentes_conectados(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    cv2.imshow('Imagen Binarizada', thresh)

    suma_total = 0  # Inicializar suma total de billetes

    for i in range(1, num_labels):  # Ignorar fondo
        x, y, w, h, area = stats[i]
        if 200 < w < 500 and 200 < h < 500:
            subimg = frame[y:y+h, x:x+w]
            subimg = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
            subimg = extraer_caracteristicas(subimg)
            subimg = scaler.transform(subimg)
            etiqueta = modelo.predict(subimg)[0]

            # Sumar el valor del billete
            suma_total += etiqueta

            # Dibujar en pantalla
            cv2.putText(frame, f"${etiqueta}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar la suma total en pantalla
    cv2.putText(frame, f"Total: ${suma_total}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el cuadro.")
            break

        obtener_componentes_conectados(frame)
        cv2.imshow('Video en vivo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()