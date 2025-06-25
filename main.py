from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import io
import base64
import time
import numpy as np
from PIL import Image
import insightface

app = FastAPI()

@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"})

from fastapi.middleware.cors import CORSMiddleware

# Allow local dev frontend
origins = [
    "http://localhost:5173",       # Vite dev server
   # "https://your-app-name.bolt.ai",  # Your deployed frontend, add when needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # You can also use ["*"] for testing only (not recommended for prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


face_model = insightface.app.FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
face_model.prepare(ctx_id=0)

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
        faces = face_model.get(np_img)

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            cropped = np_img[y1:y2, x1:x2]
            cropped_pil = Image.fromarray(cropped)
            buffer = io.BytesIO()
            cropped_pil.save(buffer, format="JPEG")
            base64_cropped = base64.b64encode(buffer.getvalue()).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{base64_cropped}"

            all_faces.append({
                "id": f"face_{face_id_counter}",
                "imageId": f"img_{image_id_counter}",
                "boundingBox": {
                    "topLeft": [x1, y1],
                    "bottomRight": [x2, y2],
                    "probability": round(face.det_score, 4)  # optional, available in InsightFace
                },
                "embedding": face.embedding.tolist(),
                "croppedFace": data_url
            })

            face_id_counter += 1

        image_id_counter += 1

    end_time = time.time()

    return JSONResponse(content={
        "success": True,
        "faces": all_faces,
        "totalFaces": len(all_faces),
        "processingTime": int((end_time - start_time) * 1000)  # in ms
    })
