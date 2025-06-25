from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import io
import base64
import time
import numpy as np
from PIL import Image
import insightface
import logging
import gc

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Setup FastAPI app
app = FastAPI()

# CORS for local/dev frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with ["http://localhost:5173"] for stricter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Server info
@app.get("/info")
async def info():
    return {
        "model": "InsightFace - buffalo_s",
        "version": "0.1",
        "max_batch": 50,
        "recommend_downscale": "1024px @ 85% JPEG"
    }

# Load InsightFace model
face_model = insightface.app.FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
face_model.prepare(ctx_id=0)

@app.post("/process-images")
async def process_images(files: List[UploadFile] = File(...)):
    start_time = time.time()
    all_faces = []
    face_id_counter = 0

    for image_id, file in enumerate(files):
        try:
            image_bytes = await file.read()
            with Image.open(io.BytesIO(image_bytes)) as img:
                img = img.convert("RGB")
                np_img = np.array(img)

            faces = face_model.get(np_img)
            logging.info(f"🖼️ Processed {file.filename} — {len(faces)} faces found")

            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                cropped = np_img[y1:y2, x1:x2]

                cropped_pil = Image.fromarray(cropped)
                buffer = io.BytesIO()
                cropped_pil.save(buffer, format="JPEG", quality=85)
                base64_cropped = base64.b64encode(buffer.getvalue()).decode("utf-8")
                data_url = f"data:image/jpeg;base64,{base64_cropped}"

                all_faces.append({
                    "id": f"face_{face_id_counter}",
                    "imageId": f"img_{image_id}",
                    "boundingBox": {
                        "topLeft": [x1, y1],
                        "bottomRight": [x2, y2],
                        "probability": round(float(face.det_score), 4)
                    },
                    "embedding": [float(x) for x in face.embedding],
                    "croppedFace": data_url
                })

                face_id_counter += 1

            # Free memory manually after each image
            del np_img, faces, img, cropped_pil, buffer, cropped
            gc.collect()

        except Exception as e:
            logging.warning(f"❌ Failed to process {file.filename}: {str(e)}")
            continue

    duration = int((time.time() - start_time) * 1000)
    logging.info(f"✅ Done. Total faces: {len(all_faces)} in {duration}ms")

    return JSONResponse(content={
        "success": True,
        "faces": all_faces,
        "totalFaces": len(all_faces),
        "processingTime": duration
    })
