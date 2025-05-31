import cv2
import numpy as np
import pickle
import os

# Obtener la ruta absoluta del archivo modelo_billetes.pkl
ruta_modelo = os.path.join(os.path.dirname(__file__), "modelo_billetes.pkl")

from skimage.feature import hog

def extraer_caracteristicas(img):
    img = cv2.resize(img, (128, 128))  # Redimensionar al mismo tamaño que en el entrenamiento
    hog_features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),  cells_per_block=(2, 2), visualize=True)
    return np.array(hog_features).reshape(1, -1)  # Ajustar la forma para la predicción


# Cargar el modelo previamente entrenado
with open(ruta_modelo, "rb") as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

def obtener_componentes_conectados(frame):
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral adaptativo para mejorar la detección
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Obtener componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    cv2.imshow('Imagen Binarizada', thresh)
    
    for i in range(1, num_labels):  # Ignorar el fondo (etiqueta 0)
        x, y, w, h, area = stats[i]
        if 200 < w < 500 and 200 < h < 500:  # Filtrar componentes por tamaño
            # Extraer subimagen del componente conectado
            subimg = frame[y:y+h, x:x+w]
            subimg = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
            subimg = cv2.resize(subimg, (90, 90))  # Resize to 90x90 pixels
            subimg = extraer_caracteristicas(subimg)
            
            # Predecir la etiqueta del billete
            etiqueta = modelo.predict(subimg)[0]
            
            # Dibujar rectángulo y etiqueta en el cuadro original
            cv2.putText(frame, f"${etiqueta}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
else:
    while True:
        # Capturar cuadro por cuadro
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el cuadro.")
            break
        
        # Procesar el cuadro para detectar billetes
        obtener_componentes_conectados(frame)
        
        # Mostrar el cuadro procesado
        cv2.imshow('Video en vivo', frame)
        
        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()