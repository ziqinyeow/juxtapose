import os
import json
from pathlib import Path
from juxtapose import RTM
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse


from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile


app = FastAPI(title="Juxt API", docs_url="/api/docs", openapi_url="/api/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/stream")
async def stream(
    config: str = Form(...),
    file: UploadFile = File(...),
):
    config = json.loads(config)
    model = RTM(
        det=config["det"],
        tracker=config["tracker"],
        pose=config["pose"],
        captions=config["detectorPrompt"],
    )

    def _stream(model, file):
        for i, res in enumerate(
            model(file, show=False, save=False, stream=True, zones=config["zones"])
        ):
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
            return {"message": "There was an error uploading the file", "path": ""}
        finally:
            file.file.close()
        return StreamingResponse(_stream(model, temp.name))

    except Exception:
        return {"message": "There was an error processing the file", "path": ""}
