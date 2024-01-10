# from juxtapose import RTM
import sys

sys.path.insert(0, "src")

# import cv2


from juxtapose import RTM, RTMDet


# model = RTM(det="rtmdet-s", tracker="bytetrack", pose="rtmpose-l")
# model("./asset/run.png", show=False)
# model("asset/bike2.mp4")
# p = RTMPose()


import supervision as sv
import cv2
import numpy as np

bboxes = [
    [178.18336181640626, 118.73443603515625, 405.0209228515625, 526.40869140625],
    [555.400634765625, 76.04942321777344, 816.806787109375, 498.3839111328125],
    [416.07158203125, 149.1138458251953, 721.727490234375, 495.57513427734375],
]

box = sv.BoxAnnotator()
detection = sv.Detections(xyxy=np.array(bboxes))

m = RTMDet("s")
# with pro[1]:
im = cv2.imread("./asset/football.jpeg")
im = cv2.resize(im, (1024, 700))
p = m(im)
box.annotate(im, p)
print(p.xyxy)
# im, e = m.preprocess(im)
# print(im.shape)
# im = cv2.imread("./asset/run.png")
# print(im.shape)
# print(m(im).xyxy)
cv2.imshow("p", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
# bboxes = m(im)
# kpts, scores = p(im, bboxes.xyxy)
# print(kpts.shape, scores.shape)
# print("First Inference: ", pro[1].dt * 1e3 / 1)
# with pro[2]:
#     im = cv2.imread("./asset/run.png")
#     # print(m(im))
# print("Second Inference: ", pro[2].dt * 1e3 / 1)
# exit()

# cv2.imshow("name", im)
# key = cv2.waitKey(0) & 0xFF
# cv2.destroyAllWindows()
