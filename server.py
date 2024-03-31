import os
import json
import uvicorn
from pathlib import Path

from juxtapose import RTM
from juxtapose.singletap import Tapnet

# from juxtapose import Tapnet
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


@app.get("/")
def ok():
    return {"status": "ok"}


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
            model(
                file,
                show=False,
                save=False,
                stream=True,
                zones=config["zones"],
                framestamp=config["framestamp"],
            )
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


@app.post("/api/tapnetstream")
async def tapnet(
    config: str = Form(...),
    file: UploadFile = File(...),
):
    config = json.loads(config)
    model = Tapnet(config["points"])

    def _stream(model, file):
        for i, res in enumerate(
            model(
                file,
                show=False,
                save=False,
                stream=True,
                # zones=config["zones"],
                # framestamp=config["framestamp"],
            )
        ):
            yield json.dumps({"frame": i, "tracks": res.tracks})
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


if __name__ == "__main__":
    print("running ok on localhost:8000")
    uvicorn.run(app, host="localhost", port=8000)

# @app.websocket("/api/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         try:
#             global global_file
#             print("websocket accepted")
#             path = await websocket.receive_text()
#             if path == global_file:
#                 print(path)
#                 for i, res in enumerate(
#                     global_model(global_file, show=False, save=False, stream=True)
#                 ):
#                     # print("Sending websocket", i)
#                     await websocket.send_json(
#                         {"frame": i, "bboxes": res.bboxes, "kpts": res.kpts}
#                     )
#                 os.remove(global_file)
#                 global_file = ""
#         except Exception as e:
#             await websocket.send_json({"success": False, "error": str(e)})
#             # await websocket.send_text(f"Message text was: error")


# @app.websocket("/api/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         try:
#             print("websocket accepted")
#             ids = await websocket.receive_text()
#             print("ids", ids)
#             b = await websocket.receive_bytes()
#             data = np.frombuffer(b, dtype=np.uint8)
#             img = cv2.imdecode(data, 1)
#             output = global_model(img, show=False)[0]  # .model_dump(exclude="im")
#             await websocket.send_json(
#                 {"id": ids, "bboxes": output.bboxes, "kpts": output.kpts}
#             )
#         except Exception as e:
#             print(e)
#             await websocket.send_json({"success": False})
#             # await websocket.send_text(f"Message text was: error")
