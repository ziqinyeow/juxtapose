import cv2


class Colors:
    """Ultralytics color palette https://ultralytics.com/."""

    def __init__(self):
        # fmt: off
        self.palette = [(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0), (255, 153, 255),
                            (153, 204, 255), (255, 102, 255), (255, 51, 255), (102, 178, 255), (51, 153, 255),
                            (255, 153, 153), (255, 102, 102), (255, 51, 51), (153, 255, 153), (102, 255, 102),
                            (51, 255, 51), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 255)]
        self.palette_index = {i: c for i, c in enumerate(self.palette)}
        # fmt: on

    def __call__(self, indexes):
        """Converts hex color codes to rgb values."""
        return [self.palette_index[idx] for idx in indexes]


# fmt: off
colors = Colors()
kpt_color = colors([16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9])
skeleton_color = colors([9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16])

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
# fmt: on


class Annotator:
    """A 17 keypoints-based annotator used to define standard styles"""

    def __init__(
        self,
        thickness: int = 1,
        kpts_thickness: int = 4,
        font=cv2.FONT_HERSHEY_DUPLEX,
        font_scale: int = 0.7,
        font_color=(255, 255, 255),
    ) -> None:
        self.thickness = thickness
        self.kpts_thickness = kpts_thickness
        self.font = font
        self.font_scale = font_scale
        self.font_color = font_color

    def draw_bboxes(self, frame, bboxes, labels=[], colors=(0, 0, 255), thickness=None):
        """Draw the bounding boxes (cv2.rectangle)
        bboxes -> [[[x1, y1, x2, y2]], ...]
        thickness -> overrides the self.thickness (int)
        """
        for i, [x1, y1, x2, y2] in enumerate(bboxes):  # index the inner element
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if len(labels) > i and labels[i]:
                self.draw_text(
                    frame, (x1 + 10, int(y1 - self.font_scale * 20)), str(labels[i])
                )
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                colors,
                thickness if thickness else self.thickness,
            )

    def draw_text(self, frame, coord, text=""):
        cv2.putText(frame, text, coord, self.font, self.font_scale, self.font_color)

    def draw_kpts(self, frame, kpts, colors=kpt_color, thickness=None):
        """Plot the 17 keypoints (cv2.circle)
        kpts -> [[[x, y] * 17], ...] -> shape: (number of human, 17, 2)
        colors -> [(0, 255, 0), ...] -> shape: (17, )
        thickness -> overrides the self.thickness (int)
        """
        for kpt in kpts:
            for i, k in enumerate(kpt):
                cv2.circle(
                    frame,
                    (int(k[0]), int(k[1])),
                    thickness if thickness else self.kpts_thickness,
                    colors[i],
                    -1,
                )

    def draw_skeletons(self, frame, kpts, colors=skeleton_color, thickness=None):
        """Plot the skeletons between two keypoints (cv2.line)
        kpts -> [[[x, y] * 17], ...] -> shape: (number of human, 17, 2)
        colors -> [(0, 255, 0), ...] -> shape: (17, )
        thickness -> overrides the self.thickness (int)
        """

        for kpt in kpts:
            for i, [idx1, idx2] in enumerate(skeleton):
                x1, y1 = kpt[idx1 - 1]
                x2, y2 = kpt[idx2 - 1]
                cv2.line(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    colors[i],
                    thickness=thickness if thickness else self.thickness,
                )
