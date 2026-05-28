from ultralytics import YOLO

# ======================================
# CARGAR MODELO PREENTRENADO
# ======================================
# Puedes usar:
# yolov8n.pt  -> rápido y liviano
# yolov8s.pt
# yolov8m.pt
# yolov8l.pt
# yolov8x.pt

model = YOLO("yolov8n.pt")

# ======================================
# ENTRENAMIENTO
# ======================================
model.train(
    data="data/data.yaml",   # Ruta del YAML
    epochs=30,          # Número de épocas
    imgsz=640,          # Tamaño imágenes
    batch=2,            # Batch pequeño para pruebas
    name="rubik_model"  # Nombre del entrenamiento
)

# ======================================
# MENSAJE FINAL
# ======================================
print("Entrenamiento terminado")
print("Modelo guardado en:")
print("runs/detect/rubik_model/weights/best.pt")