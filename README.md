## 🍿 Intro

Juxtapose is a 2D multi person pose detection, tracking, and estimation inference toolbox for sports + kinematics analysis. Visit [Docs](https://sdk.juxt.space).

<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://raw.githubusercontent.com/ziqinyeow/juxtapose/main/asset/juxtapose-banner.png"
      >
    </a>
  </p>
</div>

<!-- # JUXTAPOSE Inference Toolbox 🚀 with ONNX -->

**See how we integrated juxtapose into this app: [Juxt Space](https://github.com/ziqinyeow/juxt.space)**

## 🍄 Overview

Code mostly adopted from four repos -> [ultralytics](https://github.com/ultralytics/ultralytics), [mmdeploy](https://github.com/open-mmlab/mmdeploy), [mmdetection](https://github.com/open-mmlab/mmdetection), [mmpose](https://github.com/open-mmlab/mmpose).

Supported Detectors: [rtmdet-s](./rtm/detectors/rtmdet/), [rtmdet-m](./rtm/detectors/rtmdet/), [rtmdet-l](./rtm/detectors/rtmdet/), [groundingdino](./rtm/detectors/groundingdino/__init__.py), [yolov8](./rtm/detectors/yolov8/__init__.py) \
Supported Pose Estimators: [rtmpose-s](./rtm/rtmpose.py), [rtmpose-m](./rtm/rtmpose.py), [rtmpose-l](./rtm/rtmpose.py) \
Supported Trackers: [bytetrack](./rtm/trackers/byte_tracker.py), [botsort](./rtm/trackers/bot_sort.py)
Supported Point Trackers: [Tapnet](https://github.com/google-deepmind/tapnet)

## 🥒 Updates

- **`2024/05/16`** Remove ultralytics dependency, port yolov8 to run in ONNX directly to improve speed.
- **`2024/04/27`** Added FastAPI to EXE example with ONNX GPU Runtime in [examples/fastapi-pyinstaller](./examples/fastapi-pyinstaller).
- **`2024/01/11`** Added Nextra docs + deployed to Vercel at [sdk.juxt.space](https://sdk.juxt.space).
- **`2024/01/07`** Reduce dependencies by removing MMCV, MMDet, MMPose SDK, run fully on ONNX.
- **`2023/11/01`** Added juxtapose to PYPI repository so that we can install it using `pip install juxtapose`.
- **`2023/08/25`** Added custom [region of interests (ROI) drawing tools](rtm/utils/roi.py) that enables multi ROIs filtering while performing pose estimation/tracking. See [usage below](#🎨-select-region-of-interests-rois).
- **`2023/08/15`** Added [GroundingDino](https://github.com/IDEA-Research/GroundingDINO) & [YOLOv8](https://github.com/ultralytics/ultralytics) object detector.
- **`2023/08/09`** Added keypoints streaming to csv file using csv module.
- **`2023/07/31`** Added [ByteTrack](./rtm/trackers/byte_tracker.py) and [BotSORT](./rtm/trackers/bot_sort.py). Completed engineering effort for top down inferences in any sources. See [supported sources below](#supported-sources).
- **`2023/06/15`** Converted [RTMDET (s/m/l)](rtm/detectors/rtmdet/__init__.py) and [RTMPOSE (s/m/l)](rtm/rtmpose.py) to ONNX using [MMDeploy](https://github.com/open-mmlab/mmdeploy).

## 👉 Getting Started

### Install Using PIP

`pip install juxtapose`

Note: If you faced any issues, kindly review this [github issue](https://github.com/ziqinyeow/juxtapose/issues/2)

## 🧀 Local Development

```bash
git clone https://github.com/ziqinyeow/juxtapose
pip install .
```

## 🤩 Feel The Magic

### 🌄 Basic Usage

```python
from juxtapose import RTM

# Init a rtm model (including rtmdet, rtmpose, tracker)
model = RTM(
    det="rtmdet-m", # see type hinting
    pose="rtmpose-m", # see type hinting
    tracker="bytetrack", # see type hinting
    device="cpu",  # see type hinting
)

# Inference with directory (all the images and videos in the dir will get inference sequentially)
model("data")

# Inference with image
model("data/football.jpeg", verbose=False) # verbose -> disable terminal printing

# Inference with video
model("data/bike.mp4")

# Inference with the YouTube Source
model("https://www.youtube.com/watch?v=1vYvTbDJuFs&ab_channel=PeterGrant", save=True)
```

### 🎨 Select Region of Interests (ROIs)

It will first prompt the user to draw the ROIs, press `r` to remove the existing ROI drawn.
After drawing, press `SPACE` or `ENTER` or `q` to accept the ROI drawn. The model will filter
out the bounding boxes based on the ROIs.

😁 Note: Press `SPACE` again to redraw the bounding boxes. See custom implementation with `cv2` [here](rtm/utils/roi.py).

```python
from juxtapose import RTM

model = RTM(det="groundingdino", pose="rtmpose-l", tracker="none")
model("data/bike.mp4", roi="rect") # rectangle roi

# 1. Draw ROI first
# 2. Press r or R to reset ROI
# 3. Press SPACE or Enter or q or Q to continue with the ROI
```

### 🚴‍♂️ Accessing result for each frame: More Flexibility

```python
# Adding custom plot
import cv2
from juxtapose import RTM, Annotator

model = RTM()
annotator = Annotator(thickness=3, font_color=(128, 128, 128)) # see rtm.utils.plotting

# set show to true -> cv2.imshow the frame (you can use cv2 to plot anything in the frame)
# set plot to false -> if you want to ignore default plot -> see rtm.rtm (line `if plot:`)
for result in model("data/bike.mp4", show=True, plot=False, stream=True):
    # do what ever you want with the data
    im, bboxes, kpts = result.im, result.bboxes, result.kpts

    # e.g custom plot anything using cv2 API
    cv2.putText(
        im, "custom text", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128)
    )

    # use the annotator class -> see rtm.utils.plotting
    annotator.draw_bboxes(
        im, bboxes, labels=[f"children_{i}" for i in range(len(bboxes))]
    )
    annotator.draw_kpts(im, kpts, thickness=4)
    annotator.draw_skeletons(im, kpts)
```

### ⚽️ Custom Forward Pass: Full Flexibility

```python
# Custom model forward pass
import cv2
import torch
from juxtapose import RTMDet, RTMPose, Annotator

frame = cv2.imread("data/football.jpeg")
device = "cuda" if torch.cuda.is_available() else "cpu"

# s, m, l
rtmdet = RTMDet("l", device=device)
rtmpose = RTMPose("l", device=device)
annotator = Annotator()


bboxes, scores, labels = rtmdet(frame)  # [[x1, y1, x2, y2], ...], [], []
kpts = rtmpose(frame, bboxes=bboxes)  # shape: (number of human, 17, 2)

annotator.draw_bboxes(frame, bboxes, labels=[f"person_{i}" for i in range(len(bboxes))])
annotator.draw_kpts(frame, kpts, thickness=4)
annotator.draw_skeletons(frame, kpts)

cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Supported Sources

Adopted from ultralytics repository -> see [https://docs.ultralytics.com/modes/predict/](https://docs.ultralytics.com/modes/predict/)

| Source     | Argument                                 | Type                              | Notes                                                                     |
| ---------- | ---------------------------------------- | --------------------------------- | ------------------------------------------------------------------------- |
| image      | 'image.jpg'                              | str or Path                       | Single image file.                                                        |
| URL        | 'https://ultralytics.com/images/bus.jpg' | str                               | URL to an image.                                                          |
| screenshot | 'screen'                                 | str                               | Capture a screenshot.                                                     |
| PIL        | Image.open('im.jpg')                     | PIL.Image                         | HWC format with RGB channels.                                             |
| OpenCV     | cv2.imread('im.jpg')                     | np.ndarray of uint8 (0-255)       | HWC format with BGR channels.                                             |
| numpy      | np.zeros((640,1280,3))                   | np.ndarray of uint8 (0-255)       | HWC format with BGR channels.                                             |
| torch      | torch.zeros(16,3,320,640)                | torch.Tensor of float32 (0.0-1.0) | BCHW format with RGB channels.                                            |
| CSV        | 'sources.csv'                            | str or Path                       | CSV file containing paths to images, videos, or directories.              |
| video      | 'video.mp4'                              | str or Path                       | Video file in formats like MP4, AVI, etc.                                 |
| directory  | 'path/'                                  | str or Path                       | Path to a directory containing images or videos.                          |
| glob       | 'path/\*.jpg'                            | str                               | Glob pattern to match multiple files. Use the \* character as a wildcard. |
| YouTube    | 'https://youtu.be/Zgi9g1ksQHc'           | str                               | URL to a YouTube video.                                                   |
| stream     | 'rtsp://example.com/media.mp4'           | str                               | URL for streaming protocols such as RTSP, RTMP, or an IP address.         |
