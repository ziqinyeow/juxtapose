# RTM Inference Toolbox 🚀 with RTMDet & RTMPose & Tracker (ONNXRuntime)

## 🫰 Overview

Code mostly adopted from four repos -> [ultralytics](https://github.com/ultralytics/ultralytics), [mmdeploy](https://github.com/open-mmlab/mmdeploy), [mmdetection](https://github.com/open-mmlab/mmdetection), [mmpose](https://github.com/open-mmlab/mmpose).

Supported Detectors: [rtmdet-s](./rtm/model/rtmdet-s/), [rtmdet-m](./rtm/model/rtmdet-m/), [rtmdet-l](./rtm/model/rtmdet-l/) \
Supported Pose Estimators: [rtmpose-s](./rtm/model/rtmpose-s/), [rtmpose-m](./rtm/model/rtmpose-m/), [rtmpose-l](./rtm/model/rtmpose-l/) \
Supported Trackers: [bytetrack](./rtm/trackers/byte_tracker.py), [botsort](./rtm/trackers/bot_sort.py)

## 👉 Getting Started

### Mac (CPU only)

```bash
git clone https://github.com/ziqinyeow/rtm
cd rtm
pip install -r requirements.txt

```

### Windows (CPU & CUDA)

```bash
git clone https://github.com/ziqinyeow/rtm
cd rtm
pip install -r requirements.txt
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

```

## 🤩 Feel The Magic

### Basic Usage

```python
from rtm import RTM

# Init a rtm model (including rtmdet, rtmpose, tracker)
model = RTM(
    rtmdet="s" | "m" | "l",  # choose 1
    rtmpose="s" | "m" | "l",  # choose 1
    tracker="bytetrack" | "botsort",  # choose 1
    device="cpu" | "cuda",  # choose 1
)

# Inference with directory (all the images and videos in the dir will get inference)
model("data")

# Inference with image
model("data/football.jpeg", verbose=False) # verbose -> disable terminal printing

# Inference with video
model("data/bike.mp4")

# Inference with the YouTube Source
model("https://www.youtube.com/watch?v=1vYvTbDJuFs&ab_channel=PeterGrant", save=True)
```

### Accessing result for each frame: More Flexibility

```python
# Adding custom plot
import cv2
from rtm import RTM, Annotator

model = RTM()
annotator = Annotator(thickness=3, font_color=(128, 128, 128)) # see rtm.utils.plotting

# set show to true -> cv2.imshow the frame (you can use cv2 to plot anything in the frame)
# set plot to false -> if you want to ignore default plot -> see rtm.rtm (line `if plot:`)
for result in model("data/football.jpeg", show=True, plot=False, stream=True):
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

### Custom Forward Pass: Full Flexibility

```python
# Custom model forward pass
import cv2
import torch
from rtm import RTMDet, RTMPose, Annotator

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
