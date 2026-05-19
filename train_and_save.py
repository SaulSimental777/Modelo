"""
Script de entrenamiento - ejecutar UNA VEZ localmente antes de deployar.
Guarda el modelo entrenado en saved_model/comic_recommender.keras

Uso:
    python train_and_save.py
"""

import os
import keras
import numpy as np
import tensorflow as tf
from modelClass import RetrievalModel
from flask_pymongo import pymongo
from dotenv import load_dotenv

load_dotenv()

print("Conectando a MongoDB...")
CONNECTION_STRING = os.environ.get("MONGO_URL")
client = pymongo.MongoClient(CONNECTION_STRING)
db = client.get_database('test')

interactions = list(db.interactions.find({}))
comics = list(db.products.find({}))

df_interactions = [(k['user_id'], k['product_id'], k['score']) for k in interactions]
df_comics = [(j['productModel_id'], j['title']) for j in comics]

print(f"Interacciones: {len(df_interactions)}, Comics: {len(df_comics)}")

def preprocess_rating(row):
    user_id, comic_id, score = row
    return (
        tf.strings.to_number(str(user_id), out_type=tf.int32),
        {
            "comic_id": tf.strings.to_number(str(comic_id), out_type=tf.int32),
            "score": tf.cast(score, dtype=tf.float32)
        }
    )

processed_interactions = [preprocess_rating(row) for row in df_interactions]

shuffled_interactions = tf.data.Dataset.from_generator(
    lambda: processed_interactions,
    output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.int32),
        {
            "comic_id": tf.TensorSpec(shape=(), dtype=tf.int32),
            "score": tf.TensorSpec(shape=(), dtype=tf.float32)
        }
    )
).shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train_interactions = shuffled_interactions.take(int(0.8 * len(processed_interactions)))
test_interactions = shuffled_interactions.skip(int(0.8 * len(processed_interactions)))

train_interactions = train_interactions.batch(100).cache()
test_interactions = test_interactions.batch(100).cache()

user_ids = {row[0] for row in df_interactions}
comic_ids = {row[1] for row in df_interactions}
users_count = len(user_ids)
comics_count = len(comic_ids)

print(f"Usuarios únicos: {users_count}, Comics únicos: {comics_count}")
print("Creando y entrenando modelo...")

model = RetrievalModel(users_count + 1, comics_count + 1)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1))
model.fit(train_interactions, epochs=10)

print("Evaluando modelo...")
model.evaluate(test_interactions)

# Guardar el modelo
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/comic_recommender.keras")
print("✅ Modelo guardado en saved_model/comic_recommender.keras")

# Guardar también los metadatos (mapeo id -> título, conteos)
import json
metadata = {
    "users_count": users_count,
    "comics_count": comics_count,
    "comic_id_to_title": {str(int(str(row[0]))): row[1] for row in df_comics}
}
with open("saved_model/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False)
print("✅ Metadatos guardados en saved_model/metadata.json")
