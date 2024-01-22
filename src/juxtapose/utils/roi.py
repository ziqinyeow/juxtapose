import cv2
import supervision as sv
from typing import Literal
from juxtapose.utils import LOGGER


def select_roi(
    im,
    win: str = "roi",
    type: Literal["rect"] = "rect",
    color=sv.ColorPalette.default(),
):
    x0, y0 = -1, -1
    points, bboxes = [], []
    img, img4show = im.copy(), im.copy()
    col = ()

    def BOX(event, x, y, flags, param):
        nonlocal x0, y0, img4show, img, col
        if event == cv2.EVENT_LBUTTONDOWN:
            x0, y0 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            img4show = img.copy()
            col = color.by_idx(len(bboxes))
            cv2.rectangle(img4show, (x0, y0), (x, y), col.as_bgr(), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            img = img4show
            bboxes.append([x0, y0, x, y])

    def POINT(event, x, y, flags, param):
        nonlocal x0, y0, img4show, img
        if event == cv2.EVENT_LBUTTONDOWN:
            x0, y0 = x, y
            points.append([x0, y0])
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.circle(img4show, (x0, y0), 4, color.by_idx(len(points)).as_bgr(), 6)
            img = img4show

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, BOX if type == "rect" else POINT)

    # display the window
    while True:
        cv2.putText(
            img4show,
            "SELECT ROI",
            (200, 200),
            cv2.FONT_HERSHEY_DUPLEX,
            2,
            (255, 255, 0),
        )
        cv2.imshow(win, img4show)
        k = cv2.waitKey(1) & 0xFF
        if k == 114 or k == 82:  # RESET with 'r' or 'R'
            img = im.copy()
            img4show = im.copy()
            points = []
            bboxes = []
        elif k == 113 or k == 81 or k == 27:  # return none with 'q' or 'Q' or 'ESC'
            # return [], []
            break
        elif k == 32 or k == 13:  # DONE with 'SPACE' or 'ENTER'
            break
        elif k != 255:
            LOGGER.info(
                """After selected the ROI,
Press SPACE/ENTER     to continue
Press r/R             to restart
Press q/Q/ESC         to quit
                """
            )

    cv2.destroyWindow(win)

    if type == "rect":
        return bboxes
    else:
        return points, [i + 1 for i in range(len(points))]
