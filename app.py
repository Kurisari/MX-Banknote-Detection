from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import pickle
from skimage.feature import hog
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar modelo y scaler
with open("modelo_completo.pkl", "rb") as f:
    data = pickle.load(f)
    modelo = data["modelo"]
    scaler = data["scaler"]
    

def extraer_hog(img):
    img = cv2.resize(img, (128, 128))
    features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features

def detectar_billetes(imagen_path):
    imagen = cv2.imread(imagen_path)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, umbral = cv2.threshold(imagen_gris, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(umbral, connectivity=8)
    boxes = []

    # Filtrar componentes conectados razonables
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        print(f"Componente {i}: w={w}, h={h}, área={area}")
        if 50 < w < 500 and 30 < h < 300:
            boxes.append((x, y, w, h))

    # Eliminar cajas contenidas dentro de otras
    def is_inside(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return bx <= ax and by <= ay and (ax + aw) <= (bx + bw) and (ay + ah) <= (by + bh)

    final_boxes = []
    for i, box in enumerate(boxes):
        if not any(is_inside(box, other) for j, other in enumerate(boxes) if i != j):
            final_boxes.append(box)

    total = 0
    etiquetas = []

    for i, (x, y, w, h) in enumerate(final_boxes):
        subimg = imagen[y:y+h, x:x+w]
        subimg_gray = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
        caracteristicas = extraer_hog(subimg_gray)
        caracteristicas = scaler.transform([caracteristicas])  # Normaliza antes de predecir
        etiqueta = modelo.predict(caracteristicas)[0]
        etiquetas.append(etiqueta)
        total += int(etiqueta)

        # Dibujar
        cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)
        texto_y = max(y - 10, 20)
        cv2.putText(imagen, f"${etiqueta}", (x, texto_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    resultado_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resultado.jpeg')
    cv2.imwrite(resultado_path, imagen)

    return total, resultado_path, etiquetas

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'imagen' not in request.files:
            return "No se subió ninguna imagen"
        archivo = request.files['imagen']
        if archivo.filename == '':
            return "Nombre de archivo vacío"

        path_guardado = os.path.join(app.config['UPLOAD_FOLDER'], archivo.filename)
        archivo.save(path_guardado)

        total, resultado_path, etiquetas = detectar_billetes(path_guardado)
        return render_template('index.html', total=total, imagen=resultado_path, etiquetas=etiquetas)

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)