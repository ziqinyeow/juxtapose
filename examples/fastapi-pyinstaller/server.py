import os
import sys
import time

start = time.time()

sys.path.insert(0, "src")
sys.path.insert(1, "examples/fastapi-pyinstaller")

import json
import uvicorn
from pathlib import Path
from typing import Dict

import numpy as np
from juxtapose import RTM, RTMPose
from juxtapose.singletap import Tapnet
from juxtapose.detectors import get_detector

from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import StreamingResponse
from juxtematics.human_profile import HumanProfile
from juxtematics.constants import BODY_JOINTS_MAP
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import onnxruntime as ort

importing_time = time.time()

port = 1421

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
    return {"status": "ok", "gpu": ort.get_device() == "GPU"}


@app.get("/dir")
def dir():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    return {"dir": dirname, "file": filename}


@app.get("/model")
def model():
    path = "model"
    has_model = os.path.isdir(path)
    if has_model:
        model_dir = [p.split(".")[0].split("_")[0] for p in os.listdir(path)]
    else:
        model_dir = []
    return {"has_model": has_model, "model_dir": model_dir}


@app.post("/model/download")
def download_model(
    type: str = Body(..., embed=True), name: str = Body(..., embed=True)
):
    if type == "detector":
        get_detector(name, captions="")
    elif type == "pose":
        RTMPose(name.split("-")[1])
    elif type == "tapnet":
        Tapnet()

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

    def _stream(model: Tapnet, file):
        for i, res in enumerate(
            model(
                file,
                show=False,
                save=False,
                stream=True,
                startFrame=config["startFrame"],
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


def get_valid_joint_name(joint_name):
    if joint_name in BODY_JOINTS_MAP:
        return BODY_JOINTS_MAP[joint_name]
    return joint_name


@app.post("/api/humans")
def humans(
    humans: Dict,
    preprocess_interpolate: bool = False,
    preprocess_filter: bool = False,
    preprocess_smoothing: bool = False,
    postcalculate_filter: bool = False,
    postcalculate_smoothing: bool = False,
):
    humans = humans["humans"]

    # Humans is array of {id: 1, body_joints: [[[1,2],[3,4],...]]}
    result_humans = []
    for individual_human in humans:
        if individual_human["id"] == "":
            continue
        human_profile = HumanProfile(human_idx=int(individual_human["id"]))
        human_profile.init_with_data(np.array(individual_human["body_joints"]))
        human_profile.compute(
            preprocess_interpolate_on=preprocess_interpolate,
            preprocess_filter_on=preprocess_filter,
            preprocess_smoothing_on=preprocess_smoothing,
            postcalculate_filter_on=postcalculate_filter,
            postcalculate_smoothing_on=postcalculate_smoothing,
        )
        metrics = human_profile.get_metrics()
        result_humans.append({"id": individual_human["id"], "metrics": metrics})
        # human_profile.export_csv("output")

    # SPECIAL PROCESSING JUST FOR FRONTEND
    output_array = []

    for entry in result_humans:
        id_value = entry["id"]
        metrics = entry["metrics"]
        output_object = {}

        for metric_category, body_part_data in metrics.items():
            if metric_category not in ["body_joints_metrics", "custom_metrics"]:
                continue  # Skip irrelevant keys

            for body_part, metric_data in body_part_data.items():
                # Create metric if not exists
                for metric_name, data in metric_data.items():
                    if metric_name not in output_object:
                        output_object[metric_name] = {
                            "name": metric_name,
                            "body_joint": [],
                        }
                    # If metric exists, append data into the existing metric
                    output_object[metric_name]["body_joint"].append(
                        {
                            "name": (
                                get_valid_joint_name(body_part)
                                if metric_category == "body_joints_metrics"
                                else body_part
                            ),
                            "data": data,
                            "type": "line",
                        }
                    )

        output_array.append({"id": id_value, "transformedMetrics": output_object})
    # export output_array
    # with open("output.json", "w") as outfile:
    #     json.dump(output_array, outfile)
    return {"status": "ok", "results": output_array}


@app.post("/api/human")
def human(
    human: Dict,
    preprocess_interpolate_on=False,
    preprocess_filter_on=False,
    preprocess_smoothing_on=True,
    postcalculate_filter_on=True,
    postcalculate_smoothing_on=True,
):
    human = human["human"]
    human_profile = HumanProfile()
    human_profile.init_with_data(np.array(human["body_joints"]))
    human_profile.compute(
        preprocess_interpolate_on=preprocess_interpolate_on,
        preprocess_filter_on=preprocess_filter_on,
        preprocess_smoothing_on=preprocess_smoothing_on,
        postcalculate_filter_on=postcalculate_filter_on,
        postcalculate_smoothing_on=postcalculate_smoothing_on,
    )
    metrics = human_profile.get_metrics()
    return {"status": "ok", "results": metrics}


if __name__ == "__main__":
    total_time = time.time()
    print(f"running ok on localhost:{port}")
    print(
        f"loaded package in: {importing_time - start}, total time used: {total_time - start}"
    )
    uvicorn.run(app, host="localhost", port=port, reload=False)
