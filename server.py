import sys
from pathlib import Path

import cv2
import asyncio
import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from rtm import RTM

model = RTM(det="rtmdet-m", tracker="bytetrack", pose="rtmpose-m")


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    print(file.filename)
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            bytes = await websocket.receive_bytes()
            data = np.frombuffer(bytes, dtype=np.uint8)
            img = cv2.imdecode(data, 1)
            output = model(img, show=False)[0].model_dump(exclude="im")
            # print(output)
            await websocket.send_json(output)
        except Exception as e:
            print(e)
            await websocket.send_json({"success": False})
            # await websocket.send_text(f"Message text was: error")
