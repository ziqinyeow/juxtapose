from typing import Any, Optional
from rtm.rtm import Result
from .constants import BODY_JOINTS_MAP
from .derivatives import calculate_angle

# setup logger
from rtm.utils import LOGGER
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class Record:
    def __init__(self, *args: Any, **kwds: Any) -> Any:
        self.frequency = None

        pass

    def calc(self) -> Any:
        pass

    def export(self):
        pass


class HumanProfile(Record):
    def __init__(self, num_frames, save_dirs=None) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.body_joints = {
            i: np.full((num_frames, 2), np.nan)
            for i in range(17)
            # i: [[]] * num_frames for i in range(0, 17)
        }

        # Initialize with empty lists for each joint
        ## {
        ## 0: [[x, y], [x, y], [x, y], ...num_frames],
        ## 1: [[x, y], [x, y], [x, y], ...num_frames],
        ## ...
        ## 16: [[x, y], [x, y], [x, y], ...num_frames],
        # }
        self.df = None
        self.save_dirs = None

    def appendFrame(self, points: list[list[float]], frame_no) -> None:
        """

        Args:
            points (list[list[float]]): shape -> (17, 2)
        """
        for index, point in enumerate(points):
            self.body_joints[index][
                frame_no
            ] = point  # append the coordinates of each joint as [x, y], but will have empty points

    def __repr__(self) -> str:
        info = f"HumanProfile: {self.body_joints}\n"
        # info = f"Length: {len(self.body_joints[0])}\n Real Length: {self.num_frames}\n"
        return info

    def calculate(self):
        # Create separate DataFrames for X and Y coordinates
        df_x = pd.DataFrame({k: v[:, 0] for k, v in self.body_joints.items()})
        df_y = pd.DataFrame({k: v[:, 1] for k, v in self.body_joints.items()})

        # Add MultiIndex to the columns
        df_x.columns = pd.MultiIndex.from_product(
            [df_x.columns, ["X"]], names=["Keypoint", "Coordinate"]
        )
        df_y.columns = pd.MultiIndex.from_product(
            [df_y.columns, ["Y"]], names=["Keypoint", "Coordinate"]
        )

        # Concatenate X and Y DataFrames
        df = pd.concat([df_x, df_y], axis=1)

        # Set the index name
        df.index.name = "Frame"

        # Define the indices of the body joints (left hip, left knee, left ankle)
        joint_indices = [11, 13, 15]

        # Create an empty list to store angle values
        angles = []

        # Iterate over rows in the DataFrame
        for index, row in df.iterrows():
            # Extract the coordinates of the specified body joints
            points = np.array([[row[(i, "X")], row[(i, "Y")]] for i in joint_indices])

            # Calculate the angle between the specified body joints
            angle = calculate_angle(*points)

            # Append the angle to the list
            angles.append(angle)

        # Create a new column 'left_angle' in the DataFrame
        df["left_angle"] = angles
        self.df = df

    def calculate_and_save(self, save_dirs=None, human=None):
        if save_dirs is None:
            return

        human_folder = os.path.join(save_dirs, f"human_{human}")
        os.makedirs(human_folder, exist_ok=True)

        # Calculate angular velocity for left angle
        left_angle_values = self.df["left_angle"].values
        left_angle_diff = np.diff(left_angle_values)
        left_angular_velocity = np.concatenate(
            ([0], left_angle_diff)
        )  # Add a 0 for the first frame

        for keypoint_idx in range(17):
            keypoint_folder = os.path.join(
                human_folder, f"{keypoint_idx}_{BODY_JOINTS_MAP[keypoint_idx]}"
            )
            os.makedirs(keypoint_folder, exist_ok=True)

            df_x = self.df[keypoint_idx, "X"]
            df_y = self.df[keypoint_idx, "Y"]
            euclidean_distance = np.sqrt(df_x**2 + df_y**2)
            x_speed = df_x.diff()
            y_speed = df_y.diff()
            euclidean_speed = np.sqrt(x_speed**2 + y_speed**2)

            metrics = {
                "X Coordinate": df_x,
                "Y Coordinate": df_y,
                "Euclidean Distance": euclidean_distance,
                "X Speed": x_speed,
                "Y Speed": y_speed,
                "Speed": euclidean_speed,
            }

            for metric_name, metric_data in metrics.items():
                chart_path = os.path.join(
                    keypoint_folder, f'{metric_name.lower().replace(" ", "_")}.png'
                )

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(
                    np.arange(len(metric_data.index)), metric_data.values, color="blue"
                )
                ax.set_xlabel("Frame")
                ax.set_ylabel("Value")
                ax.set_title(f"{metric_name} (Keypoint {keypoint_idx})")
                chart_path = os.path.join(
                    keypoint_folder, f'{metric_name.lower().replace(" ", "_")}.png'
                )
                plt.savefig(chart_path)
                plt.close()

        # Save left angle and angular velocity at the same folder level as a keypoint
        left_angle_folder = os.path.join(human_folder, "left_leg")
        os.makedirs(left_angle_folder, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            np.arange(len(self.df.index)), self.df["left_angle"].values, color="blue"
        )
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle")
        ax.set_title("Left Angle")
        left_angle_chart_path = os.path.join(left_angle_folder, "left_angle.png")
        plt.savefig(left_angle_chart_path)
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.arange(len(self.df.index)), left_angular_velocity, color="red")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angular Velocity")
        ax.set_title("Left Angular Velocity")
        angular_velocity_chart_path = os.path.join(
            left_angle_folder, "angular_velocity.png"
        )
        plt.savefig(angular_velocity_chart_path)
        plt.close()


class ObjectProfile(Record):
    def __init__(self) -> None:
        super().__init__()
        pass


class Kinematics:
    def __init__(
        self, results: Optional[list[Result]] = None, run_path: Optional[str] = None
    ) -> None:
        self.frequency = None
        self.profileSet: dict[str, Record] = {}
        self.file_name = None
        self.save_dirs = None
        self.num_frames = None

        if results is not None:
            self.setup(results)
        elif run_path is not None:
            self.setup_from_path(run_path)

    def setup(self, results: Optional[list[Result]]) -> None:
        num_frames = len(results)
        self.num_frames = num_frames
        for frame_no, result in enumerate(results):
            if self.file_name is None:
                self.file_name = result.name
            if self.save_dirs is None:
                self.save_dirs = result.save_dirs

            # create profile and update profileSet
            for index, ids in enumerate(result.track_id):
                if ids not in self.profileSet:
                    self.profileSet[ids] = HumanProfile(num_frames=num_frames)
                self.profileSet[ids].appendFrame(result.kpts[index], frame_no=frame_no)
        pass

    def setup_from_path(self, run_csv: str) -> None:
        # Get the file name without extension using pathlib library
        self.file_name = Path(run_csv).stem
        self.save_dirs = str(Path(run_csv).parent)
        with open(run_csv, "r") as csv_file:
            csv_reader = csv.reader(csv_file)

            csv_list = list(csv_reader)
            self.num_frames = len(csv_list)
            for row in csv_list:
                frame_number, keypoints_data = row
                frame_number = int(frame_number)
                # Parse the JSON string
                keypoints_data = json.loads(keypoints_data)
                self.process_frame(frame_number - 1, keypoints_data)

    def process_frame(self, frame_number, keypoints_data):
        for profile_id, profile_points in keypoints_data.items():
            if profile_id == "":
                continue
            profile_id = int(profile_id)  # Convert profile_id to float
            if profile_id not in self.profileSet:
                self.profileSet[profile_id] = HumanProfile(num_frames=self.num_frames)
            self.profileSet[profile_id].appendFrame(
                profile_points, frame_no=frame_number
            )

    def __call__(self, save=True, save_dirs=None, overwrite=False) -> Any:
        if save:
            save_dirs = self.save_dirs if save_dirs is None else save_dirs
            save_dir_check = Path(save_dirs) / "human_1"
            if save_dir_check.exists() and overwrite is False:
                overwrite_prompt = input("Overwrite? (y/n)")
                overwrite = True if overwrite_prompt == "y" else False
            if save_dir_check.exists() is False or overwrite:
                for each in self.profileSet.keys():
                    self.profileSet[each].calculate()
                    self.profileSet[each].calculate_and_save(
                        save_dirs=str(Path(self.save_dirs)), human=str(each)
                    )
