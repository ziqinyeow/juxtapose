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
    [176.63677978515625, 130.05418395996094, 405.93341064453125, 572.220458984375],
    [418.30072021484375, 163.69384765625, 725.8524169921875, 541.7325439453125],
    [559.0335693359375, 81.39617919921875, 815.6915283203125, 545.7316284179688],
]

box = sv.BoxAnnotator()
detection = sv.Detections(xyxy=np.array(bboxes))

m = RTMDet("s")
# with pro[1]:
im = cv2.imread("./asset/football.jpeg")
im = cv2.resize(im, (1024, 700))
p = m(im)
box.annotate(im, p)
# print(p.xyxy)
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
