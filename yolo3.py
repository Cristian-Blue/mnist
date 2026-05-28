from ultralytics import YOLO
import cv2

# =========================
# CARGAR MODELO ENTRENADO
# =========================
model = YOLO("runs/detect/rubik_model-2/weights/best.pt")

# =========================
# ABRIR CÁMARA
# =========================
cap = cv2.VideoCapture(0)

# Verificar si la cámara abrió correctamente
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

print("Cámara iniciada...")
print("Presiona 'q' para salir")

# =========================
# BUCLE PRINCIPAL
# =========================
while True:

    # Leer frame de la cámara
    ret, frame = cap.read()

    if not ret:
        print("Error leyendo la cámara")
        break

    # =========================
    # DETECCIÓN CON YOLO
    # =========================
    results = model(frame)

    # Dibujar detecciones
    annotated_frame = results[0].plot()

    # Mostrar resultado
    cv2.imshow("Deteccion YOLOv8", annotated_frame)

    # Salir con tecla q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CERRAR TODO
# =========================
cap.release()
cv2.destroyAllWindows()