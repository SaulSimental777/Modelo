from flask import Flask, request, jsonify
import keras
import numpy as np
import tensorflow as tf
from modelClass import RetrievalModel
from dotenv import load_dotenv
import os
import json

load_dotenv()
app = Flask(__name__)
port = int(os.getenv("PORT", 3100))

# --- Cargar metadatos ---
print("Cargando metadatos del modelo...")
with open("saved_model/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

users_count = metadata["users_count"]
comics_count = metadata["comics_count"]
comic_id_to_comic_title = {int(k): v for k, v in metadata["comic_id_to_title"].items()}
comic_id_to_comic_title[0] = ""

# --- Cargar modelo pre-entrenado (sin re-entrenar) ---
print("Cargando modelo pre-entrenado...")
model = keras.models.load_model(
    "saved_model/comic_recommender.keras",
    custom_objects={"RetrievalModel": RetrievalModel}
)
print("✅ Modelo listo.")

# --- Predicción ---
def ValuePrediction(user_id):
    predictions = model.predict(keras.ops.convert_to_tensor([user_id]))
    predictions = keras.ops.convert_to_numpy(predictions["predictions"])

    recommended_comics = []
    for comic_id in predictions[0]:
        recommended_comics.append(comic_id_to_comic_title.get(int(comic_id), "Desconocido"))

    return recommended_comics

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        user_id = int(data.get("user_id"))
        print("IDENTIFICADOR RECIBIDO:", user_id)
        recommendations = ValuePrediction(user_id)
        return jsonify({
            "status": "success",
            "user_id": user_id,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(port=port, debug=False)
