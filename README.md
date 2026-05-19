# Comic Recommender — Guía de despliegue en Railway

## Estructura del proyecto

```
comic_recommender/
├── app.py                 # Servidor Flask (carga modelo pre-entrenado)
├── modelClass.py          # Definición del modelo (cópialo de tu proyecto original)
├── train_and_save.py      # Script de entrenamiento (ejecutar una sola vez)
├── requirements.txt
├── Dockerfile
├── .env                   # Variables de entorno locales (NO subir a git)
└── saved_model/           # Generado por train_and_save.py
    ├── comic_recommender.keras
    └── metadata.json
```

---

## Paso 1 — Preparar localmente

### 1.1 Copiar `modelClass.py`
Copia tu `modelClass.py` original a esta carpeta.

### 1.2 Crear tu `.env`
```
MONGO_URL=mongodb+srv://usuario:password@cluster.mongodb.net/
PORT=3100
```

### 1.3 Instalar dependencias localmente
```bash
pip install -r requirements.txt
```

### 1.4 Entrenar y guardar el modelo (solo una vez)
```bash
python train_and_save.py
```
Esto crea la carpeta `saved_model/` con el modelo y los metadatos.
**Este paso conecta a MongoDB y tarda unos minutos.**

### 1.5 Probar localmente
```bash
python app.py
# En otra terminal:
curl -X POST http://localhost:3100/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1}'
```

---

## Paso 2 — Subir a GitHub

```bash
git init
git add .
git commit -m "Comic recommender listo para deploy"
git remote add origin https://github.com/TU_USUARIO/comic-recommender.git
git push -u origin main
```

> ⚠️ Asegúrate de que `.gitignore` incluye `.env`.
> El archivo `saved_model/` SÍ debe subirse a git (es el modelo entrenado).

---

## Paso 3 — Desplegar en Railway

1. Ir a [railway.app](https://railway.app) y crear cuenta (gratis)
2. **New Project → Deploy from GitHub repo**
3. Seleccionar tu repositorio
4. Railway detecta automáticamente el `Dockerfile` y construye la imagen

### Configurar variables de entorno en Railway
En tu proyecto → **Variables** → agregar:
```
MONGO_URL=mongodb+srv://usuario:password@cluster.mongodb.net/
```

> Railway inyecta `PORT` automáticamente, no la necesitas configurar.

### Dominio público
En Railway → **Settings → Networking → Generate Domain**
Te da una URL tipo `https://comic-recommender-production.up.railway.app`

---

## Notas importantes

- **TF en Railway**: la imagen tarda ~3-5 min en construirse la primera vez (descarga TensorFlow), luego es instantáneo.
- **Workers**: el Dockerfile usa 1 worker (`--workers 1`) porque TensorFlow ocupa mucha RAM. Con el plan gratuito de Railway tienes 512MB.
- **MongoDB**: asegúrate de que tu cluster de MongoDB Atlas permite conexiones desde `0.0.0.0/0` (cualquier IP) en Network Access.
- **El modelo NO se re-entrena en cada arranque**: solo se carga desde `saved_model/`. Si cambias los datos, corre `train_and_save.py` de nuevo y sube el nuevo modelo.
