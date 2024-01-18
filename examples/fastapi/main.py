import os
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from juxtapose import RTM

app = FastAPI(title="API", docs_url="/api/docs", openapi_url="/api/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/stream")
async def stream(
    file: UploadFile = File(...),
    config: dict = {"det": "rtmdet-m", "tracker": "None", "pose": "rtmpose-m"},
):
    model = RTM(det=config["det"], tracker=config["tracker"], pose=config["pose"])

    def _stream(file):
        for i, res in enumerate(model(file, show=False, save=False, stream=True)):
            yield json.dumps({"frame": i, "persons": res.persons})
        os.remove(file)

    try:
        try:
            suffix = Path(file.filename).suffix
            temp = NamedTemporaryFile(suffix=suffix, delete=False)
            contents = file.file.read()
            with temp as f:
                f.write(contents)
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()
        return StreamingResponse(_stream(temp.name))
    except Exception:
        return {"message": "There was an error processing the file"}
