"""Pose Viewer module for visualizing movement data in notebook."""

import cv2
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from videoreader import VideoReader

from movement import sample_data
from movement.utils.logging import logger


class MovementPoseViewer:
    """A pose viewer for movement made for notebook environments.

    Attributes:
        ds: The dataset containing pose estimation data.
        keypoints: List of available keypoints in the dataset.
        video_reader: Handles video frame extraction.
        frame_count: Total number of frames in the video.
        frame_cache: Dictionary to store cached video frames for
                     performance optimization.

    """

    def __init__(self):
        """Initialize the pose viewer.

        Initializes the pose viewer by:
        1.Loading Dataset
        2.Setting up widgets

        """
        self.ds = sample_data.fetch_dataset(
            "SLEAP_single-mouse_EPM.predictions.slp", with_video=True
        )
        logger.debug("Dataset loaded successfully")

        self.keypoints = self.ds.position.keypoints.values
        logger.debug(f"Available keypoints: {list(self.keypoints)}")

        self.video_reader = None
        self.current_frame = None
        self.frame_count = 0
        self._initialize_video()

        self.out = widgets.Output()
        self.frame_slider = widgets.IntSlider(
            min=0,
            max=self.frame_count - 1 if self.frame_count > 0 else 0,
            value=0,
            description="Frame:",
            continuous_update=False,
            layout=widgets.Layout(width="80%"),
        )

        self.play_widget = widgets.Play(
            value=0,
            min=0,
            max=self.frame_count - 1 if self.frame_count > 0 else 0,
            step=1,
            interval=1000 / self.ds.attrs.get("fps", 30),
            description="Play",
        )

        self.keypoint_checkboxes = [
            self._create_checkbox(kp) for kp in self.keypoints
        ]

        self.show_video_checkbox = widgets.Checkbox(
            value=True, description="Show Video"
        )
        self.status_label = widgets.Label(value="Ready")

        self.frame_slider.observe(self.update_plot, names="value")
        self.show_video_checkbox.observe(self.update_plot, names="value")
        self.play_widget.observe(self.update_plot, names="value")
        self.play_frame_link = widgets.jslink(
            (self.play_widget, "value"), (self.frame_slider, "value")
        )

        for cb in self.keypoint_checkboxes:
            cb.observe(self.update_plot, names="value")

        self.frame_cache = {}
        self.max_cache_size = 50

    def _create_checkbox(self, keypoint):
        """Create a checkbox widget for selecting keypoints."""
        return widgets.Checkbox(value=True, description=keypoint)

    def _initialize_video(self):
        """Initialize video reader and set frame count."""
        if not hasattr(self.ds, "video_path"):
            logger.warning("No video path found in dataset")
            return

        logger.debug(f"Opening video from: {self.ds.video_path}")
        try:
            self.video_reader = VideoReader(self.ds.video_path)
            self.frame_count = len(self.video_reader)
            self.target_fps = (
                self.video_reader.frame_rate or self.ds.attrs.get("fps", 30)
            )

            self.frame_slider.max = (
                min(self.frame_count - 1, len(self.ds.time) - 1)
                if self.frame_count > 0
                else 0
            )
            self.play_widget.max = self.frame_slider.max
            self.play_widget.interval = 1000 / self.target_fps
        except Exception as e:
            logger.error(f"Failed to open video: {e}", exc_info=True)

    def _read_frame(self, frame_idx):
        """Retrieve a video frame, utilizing caching for optimization."""
        if self.video_reader is None:
            return None

        frame_idx = max(0, min(frame_idx, self.frame_count - 1))

        if frame_idx in self.frame_cache:
            return self.frame_cache[frame_idx]

        try:
            frame = self.video_reader[frame_idx]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if len(self.frame_cache) >= self.max_cache_size:
                self.frame_cache.clear()

            self.frame_cache[frame_idx] = rgb_frame
            return rgb_frame
        except Exception as e:
            logger.error(
                f"Error reading frame {frame_idx}: {e}", exc_info=True
            )
            return None

    def _get_poses_for_frame(self, frame_idx):
        """Get pose data for the given frame index."""
        if frame_idx >= len(self.ds.time) or frame_idx < 0:
            return None

        time_val = self.ds.time[frame_idx].values
        return self.ds.position.sel(time=time_val)

    def update_plot(self, change=None):
        """Update the plot with the selected frame and keypoints."""
        frame_idx = self.frame_slider.value
        visible_keypoints = [
            cb.description for cb in self.keypoint_checkboxes if cb.value
        ]

        with self.out:
            clear_output(wait=True)
            plt.figure(figsize=(10, 6))
            ax = plt.gca()

            if (
                self.show_video_checkbox.value
                and self.video_reader is not None
            ):
                frame = self._read_frame(frame_idx)
                if frame is not None:
                    ax.imshow(frame)

            pose_data = self._get_poses_for_frame(frame_idx)
            if pose_data is not None:
                colors = plt.cm.tab10(np.linspace(0, 1, len(self.keypoints)))
                for i, kp in enumerate(self.keypoints):
                    if kp in visible_keypoints:
                        x = pose_data.sel(space="x", keypoints=kp).values
                        y = pose_data.sel(space="y", keypoints=kp).values
                        if not np.isnan(x) and not np.isnan(y):
                            ax.scatter(x, y, color=colors[i], s=50, label=kp)

            ax.set_title(
                f"Frame {frame_idx + 1}/{self.frame_count}\\n"
                f"(Time: {frame_idx / self.target_fps:.2f}s)"
            )
            ax.axis("off")

            if visible_keypoints:
                ax.legend(
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    title="Keypoints",
                )

            plt.tight_layout()
            plt.show()

    def show(self):
        """Display the interactive pose viewer in Jupyter Notebook."""
        controls = widgets.VBox(
            [
                widgets.HBox(
                    [self.play_widget, self.frame_slider, self.status_label]
                ),
                widgets.HBox(
                    [
                        self.show_video_checkbox,
                        widgets.VBox(self.keypoint_checkboxes),
                    ]
                ),
            ]
        )

        main_widget = widgets.VBox([controls, self.out])
        self.update_plot()
        return main_widget

    def cleanup(self):
        """Release resources to prevent memory leaks."""
        if hasattr(self, "play_frame_link"):
            self.play_frame_link.unlink()
        if self.video_reader is not None:
            self.video_reader.close()
        plt.close("all")
        self.frame_cache.clear()


def create_viewer():
    """Instantiate and display the pose viewer in Jupyter Notebook."""
    viewer = MovementPoseViewer()
    display(viewer.show())
    return viewer
