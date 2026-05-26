from ultralytics import YOLO
import cv2

# Cargar modelo preentrenado
model = YOLO("yolov8n.pt")

# Abrir cámara
cap = cv2.VideoCapture(0)

# Verificar cámara
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:

    # Leer frame
    ret, frame = cap.read()

    if not ret:
        break

    # Realizar detección
    results = model(frame)

    # Recorrer resultados
    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Coordenadas
            x1, y1, x2, y2 = box.xyxy[0]

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confianza
            confidence = float(box.conf[0])

            # Clase detectada
            class_id = int(box.cls[0])

            # Nombre de la clase
            class_name = model.names[class_id]

            # Solo mostrar si confianza > 50%
            if confidence > 0.5:

                # Color verde para personas
                color = (0, 255, 0)

                # Dibujar rectángulo
                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              color,
                              2)

                # Texto
                text = f"{class_name} {confidence:.2f}"

                cv2.putText(frame,
                            text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2)

    # Mostrar ventana
    cv2.imshow("Deteccion YOLO", frame)

    # Salir con tecla q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()