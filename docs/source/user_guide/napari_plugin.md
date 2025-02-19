(target-napari-plugin)=
# Viewing data in napari

You can visualise `movement` motion tracks in [napari](napari:) using our
plugin, which extends `napari`'s core functionality and provides a widget
that can be docked in the `napari` window. Currently, you can use it to
visualise 2D [poses datasets](target-poses-and-bboxes-dataset)
as points overlaid on video frames.

:::{warning}
This plugin is still in early stages of development but we are working on
ironing out the kinks. [Get in touch](target-get-in-touch)
if you find any bugs or have suggestions for improvements!
:::

## Installation

The `napari` plugin is shipped with the `movement` package starting from
version `0.1.0`. If you install `movement` via `conda`, `napari` will already
be available. If you use the `pip` installer, make sure to
install `movement` with the `[napari]` extra:

```sh
pip install movement[napari]
```

## Launch napari

Type the following command in your terminal:

```sh
napari -w movement
```

This will open the `napari` window with the `movement` widget docked on the
right-hand side, as in the the [screenshot](target-widget-screenshot) below.

In `napari`, data is typically loaded into [layers](napari:guides/layers.html),
which can be reordered and toggled for visibility in the layers list panel.
For example, keypoint data can be added as a
[points layer](napari:howtos/layers/points.html),
while image stacks (including videos) can be added as
[image layers](napari:howtos/layers/image.html).
Below, we'll explain how to do this.

## Load a background layer

Though this is not strictly necessary, it is usually informative to
view the keypoints overlaid on a background that provides
some spatial context. You can either [load the video](target-load-video)
corresponding to the poses dataset, or a [single image](target-load-frame),
e.g., a still frame derived from that video. In the following sections,
we will show you how to do both and discuss some limitations.

(target-load-video)=
### Load a video

To load a video, drag and drop the video file onto the `napari` window.
You will see a pop-up dialog asking you to select the reader.
Choose the `video` reader—corresponding to the
[`napari-video`](https://github.com/janclemenslab/napari-video)
plugin—and click `OK`. You can optionally select to remember this reader
for all files with the same extension.

`napari-video` will load the video as an image stack with a slider
at the bottom that you can use to navigate through frames.
You may also use the left and right arrow keys to navigate
frame-by-frame.

Clicking on the play button will start the video playback at a default
rate of 10 frames per second. You can adjust that by right-clicking on the
play button or by opening the `napari > Preferences` menu and changing
the `Playback frames per second` setting.

:::{admonition} Video playback limitations
:class: warning

- The video playback may freeze or stutter if you click on the slider to jump
  to a specific frame. We recommended pausing the playback before such jumps.
- `napari-video` may struggle to play videos at a high frame rate, depending
  on your hardware, the video resolution and codec. If you experience
  performance issues, such as the video freezing or skipping frames,
  try reducing the playback frames per second or fall back to
  using a [single image](target-load-frame) as a background.
:::


(target-load-frame)=
### Load an image

This usually means using a still frame extracted from the video, but in theory
you could use any image that's in the same coordinate system as the
tracking data. For example, you could use a schematic diagram of the arena,
as long as it has the same width and height as the video and is
properly aligned with the tracking data.

::: {dropdown} Extracting a still frame from a video
:color: info
:icon: info

You can use the command line tool [`ffmpeg`](https://www.ffmpeg.org/)
to extract a still frame from a video.

To extract the first frame of a video:

```sh
ffmpeg -i video.mp4 -frames:v 1 first-frame.png
```

To extract a frame at a specific time stamp (e.g. at 2 seconds):

```sh
ffmpeg -i video.mp4 -ss 00:00:02 -frames:v 1 frame-2sec.png
```
:::

To load any image into `napari`, simply drag and drop the image file into
the napari window. Alternatively, you can use the `File > Open File(s)` menu
option and select the file from the file dialog.
In any case, the image will be loaded as a single 2D frame without a slider.

## Load the poses dataset

Now you are ready to load some pose tracks over your chosen background layer.

On the right-hand side of the window you should see
an expanded `Load poses` menu. To load some pose data in napari:
1. Select the `source software` from the dropdown menu.
2. Set the `fps`  (frames per second) of the video the pose data refers to. Note this will only affect the units of the time variable shown when hovering over a keypoint. If the fps is not known, you can set it to 1, which will effectively make the time variable equal to the frame number.
3. Select the file containing the predicted poses. The path can be directly pasted or you can use the file browser button.
4. Click `Load`.

The data should be loaded into the viewer as a
[points layer](napari:howtos/layers/points.html).
By default, it is added at the top of the layer list.

::: {note}
See [supported formats](target-supported-formats) for more information on
the expected software and file formats.
:::


You will see a view similar to the one below:

(target-widget-screenshot)=

![napari widget with poses dataset loaded](../_static/napari_plugin_with_poses_as_points.png)

The predicted keypoints are represented as points, colour-coded by
keypoint ID for single-individual datasets, or by individual ID for
multi-individual datasets. These IDs can be also displayed as text
next to the points by enabling the `display text` option from the
layer controls panel.

Hovering with your mouse over a point
(with the points layer selected) will
bring up a tooltip containing the names of the individual and keypoint,
the point-wise confidence score (as predicted by the source software),
and the time in seconds (this is calculated from the frame number and
the `fps` value you provided).

Using the slider at the bottom of the window, you can move through
the frames of the dataset, and the points and video will update
in sync.

::: {admonition} Stay tuned
Though the display style of the points layer is currently fixed, we are
working on adding more customisation options in future releases, such as
enabling you to change the point size, colour, or shape.

We are also working on enabling the visualisation of
[bounding boxes datasets](target-poses-and-bboxes-dataset) in the plugin.
:::
