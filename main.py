from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import io
import base64
import time
import numpy as np
from PIL import Image
from insightface.model_zoo import get_model
import uuid

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model": "SCRFD + MobileFaceNet",
        "version": "1.0",
        "description": "Lightweight InsightFace server with custom detector/embedder pipeline"
    }

# Load models
print("Loading models...")
detector = get_model('scrfd_10g_bnkps.onnx', download=True)
detector.prepare(ctx_id=0)

embedder = get_model('mobilenet.onnx', download=True)
embedder.prepare(ctx_id=0)
print("Models loaded.")

@app.post("/process-images")
async def process_images(files: List[UploadFile] = File(...)):
    start_time = time.time()
    all_faces = []
    face_id_counter = 0
    image_id_counter = 0

    for file in files:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(img)

        bboxes, landmarks = detector.detect(np_img, max_num=0, metric='default')

        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = box[:4].astype(int)
            confidence = float(box[4])
            cropped = np_img[y1:y2, x1:x2]
            cropped_pil = Image.fromarray(cropped)

            buffer = io.BytesIO()
            cropped_pil.save(buffer, format="JPEG")
            base64_cropped = base64.b64encode(buffer.getvalue()).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{base64_cropped}"

            try:
                embedding = embedder.get(cropped)
            except Exception as e:
                print("Embedding failed for face", e)
                continue

            all_faces.append({
                "id": f"face_{face_id_counter}",
                "imageId": f"img_{image_id_counter}",
                "boundingBox": {
                    "topLeft": [int(x1), int(y1)],
                    "bottomRight": [int(x2), int(y2)],
                    "probability": round(confidence, 4)
                },
                "embedding": embedding.astype(float).tolist(),
                "croppedFace": data_url
            })
            face_id_counter += 1

        image_id_counter += 1

    end_time = time.time()
    return JSONResponse(content={
        "success": True,
        "faces": all_faces,
        "totalFaces": len(all_faces),
        "processingTime": int((end_time - start_time) * 1000)
    })
