from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# cargar modelo
model = tf.keras.models.load_model("mnist_model.keras")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # imagen desde frontend
        data = request.json["image"]

        # quitar encabezado base64
        image_data = data.split(",")[1]

        # convertir base64 -> imagen
        image = Image.open(
            io.BytesIO(base64.b64decode(image_data))
        ).convert("L")

        # redimensionar a 28x28
        image = image.resize(
            (28, 28),
            Image.Resampling.LANCZOS
        )

        # convertir a numpy
        image = np.array(image)

        # convertir a float32
        image = image.astype("float32")

        # normalizar
        image = image / 255.0

        # reshape CNN
        image = image.reshape(1, 28, 28, 1)

        debug_image = (image[0] * 255).astype(np.uint8)

        Image.fromarray(
            debug_image.squeeze()
        ).save("debug.png")
        # predicción
        prediction = model.predict(image)[0]

        print("======================")
        print("Predicciones:")
        print(prediction)
        print("======================")

        # número con mayor probabilidad
        number = np.argmax(prediction)

        # porcentaje de confianza
        confidence = float(prediction[number]) * 100

        print("Número:", number)
        print("Confianza:", confidence)

        return jsonify({
            "prediction": int(number),
            "confidence": round(confidence, 2)
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)