from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import io, base64, time
import numpy as np
from PIL import Image
import insightface

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Load MobileFaceNet model
model = insightface.app.FaceAnalysis(
    name="antelope",  # "antelope" is InsightFace's MobileFaceNet-based model
    root="./models",  # Adjust path if needed
    providers=["CPUExecutionProvider"]  # Or "CUDAExecutionProvider" if GPU available
)
model.prepare(ctx_id=0)

@app.post("/process-images")
async def process_images(files: List[UploadFile] = File(...)):
    start_time = time.time()
    all_faces = []
    face_id = 0
    image_id = 0

    for file in files:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        np_img = np.array(img)
        faces = model.get(np_img)

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            crop = np_img[y1:y2, x1:x2]
            buffer = io.BytesIO()
            Image.fromarray(crop).save(buffer, format="JPEG")
            base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

            all_faces.append({
                "id": f"face_{face_id}",
                "imageId": f"img_{image_id}",
                "boundingBox": {
                    "topLeft": [x1, y1],
                    "bottomRight": [x2, y2],
                    "probability": round(face.det_score, 4)
                },
                "embedding": face.embedding.tolist(),
                "croppedFace": f"data:image/jpeg;base64,{base64_img}"
            })
            face_id += 1

        image_id += 1

    return JSONResponse(content={
        "success": True,
        "faces": all_faces,
        "totalFaces": len(all_faces),
        "processingTime": int((time.time() - start_time) * 1000)
    })
