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
from insightface.model_zoo import get_model

app = FastAPI()

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow Vite dev server and deployed frontend
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load detector and recognizer
detector = get_model("scrfd_2.5g_bnkps", download=True)
recognizer = get_model("mobilenet", download=True)  # MobileFaceNet
@app.get("/health")
async def health_check():
    return {"status": "ok"}

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

        dets = detector.detect(np_img)
        if dets is None or len(dets) == 0:
            image_id_counter += 1
            continue

        bboxes = dets[:, :4].astype(int)
        kpss = dets[:, 5:].reshape(-1, 5, 2)
        scores = dets[:, 4]

        for i, (bbox, kps, score) in enumerate(zip(bboxes, kpss, scores)):
            face_crop = insightface.utils.face_align.norm_crop(np_img, landmark=kps)
            embedding = recognizer.get(face_crop)

            buffer = io.BytesIO()
            Image.fromarray(face_crop).save(buffer, format="JPEG")
            base64_cropped = base64.b64encode(buffer.getvalue()).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{base64_cropped}"

            all_faces.append({
                "id": f"face_{face_id_counter}",
                "imageId": f"img_{image_id_counter}",
                "boundingBox": {
                    "topLeft": bbox[:2].tolist(),
                    "bottomRight": bbox[2:].tolist(),
                    "probability": float(score)
                },
                "embedding": embedding.tolist(),
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
