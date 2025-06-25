from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import insightface
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    faces = model.get(img)
    return {
        "faces": [
            {
                "bbox": face.bbox.tolist(),
                "embedding": face.embedding.tolist(),
                "gender": face.gender,
                "age": face.age,
            }
            for face in faces
        ]
    }
