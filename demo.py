# from juxtapose import RTM
# import sys

# sys.path.insert(0, "src")

# import cv2


from juxtapose import RTM


model = RTM(det="rtmdet-m", tracker="bytetrack", pose="rtmpose-l")
model("asset/bike2.mp4", show=True)
# p = RTMPose()


# with pro[0]:
#     m = RTMDet()
# print("Loading rtmdet-m: ", pro[0].dt * 1e3 / 1)
# with pro[1]:
#     im = cv2.imread("./asset/football.jpeg")
#     bboxes = m(im)
#     kpts, scores = p(im, bboxes.xyxy)
#     print(kpts.shape, scores.shape)
#     # print(m(im))
# print("First Inference: ", pro[1].dt * 1e3 / 1)
# with pro[2]:
#     im = cv2.imread("./asset/run.png")
#     # print(m(im))
# print("Second Inference: ", pro[2].dt * 1e3 / 1)
# exit()

# cv2.imshow("name", im)
# key = cv2.waitKey(0) & 0xFF
# cv2.destroyAllWindows()
