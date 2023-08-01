from rtm import RTM

# Init a rtm model (including rtmdet, rtmpose, tracker)
model = RTM(
    rtmdet="s" | "m" | "l",  # choose 1
    rtmpose="s" | "m" | "l",  # choose 1
    tracker="bytetrack" | "botsort",  # choose 1
    device="cpu" | "cuda",  # choose 1
)

# Inference with directory
model("data")

# Inference with image
model("data/football.jpeg", verbose=False)

# Inference with image
model("data/bike.mp4")


# Inference with the YouTube Source
model("https://www.youtube.com/watch?v=1vYvTbDJuFs&ab_channel=PeterGrant", save=True)

# Add custom plot
import cv2
from rtm import RTM, Annotator

model = RTM()
annotator = Annotator(thickness=3, font_color=(128, 128, 128))

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
