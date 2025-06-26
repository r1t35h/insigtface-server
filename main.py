from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from insightface.model_zoo import get_model
import numpy as np
from PIL import Image
import base64
import io
import time
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
detector = get_model("SCRFD_10G_KPS", download=True)
embedder = get_model("arcface_mobilefacenet", download=True)
assert detector is not None, "Failed to load SCRFD"
assert embedder is not None, "Failed to load MobileFaceNet"
@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})

@app.post("/process-images")
async def process_images(files: List[UploadFile] = File(...)):
    start_time = time.time()
    all_faces = []
    face_id = 0

    for image_id, file in enumerate(files):
        img_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        np_img = np.array(pil_img)

        bboxes, kpss = detector.detect(np_img, max_num=10)
        for bbox, kps in zip(bboxes, kpss):
            x1, y1, x2, y2, score = bbox.astype(int)
            face_crop = np_img[y1:y2, x1:x2]
            emb = embedder.get(np_img, kps)

            buffer = io.BytesIO()
            Image.fromarray(face_crop).save(buffer, format="JPEG")
            cropped_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            all_faces.append({
                "id": f"face_{face_id}",
                "imageId": f"img_{image_id}",
                "boundingBox": {
                    "topLeft": [int(x1), int(y1)],
                    "bottomRight": [int(x2), int(y2)],
                    "probability": float(score)
                },
                "embedding": emb.tolist(),
                "croppedFace": f"data:image/jpeg;base64,{cropped_base64}"
            })
            face_id += 1

    end_time = time.time()

    return JSONResponse(content={
        "success": True,
        "faces": all_faces,
        "totalFaces": len(all_faces),
        "processingTime": int((end_time - start_time) * 1000)
    })
