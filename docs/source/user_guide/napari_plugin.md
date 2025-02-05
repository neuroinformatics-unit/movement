(target-napari-plugin)=
# The napari plugin

To enable the interactive visualisation of motion tracks in
`movement`, we have created an experimental plugin for [napari](napari:).
Currently, the plugin supports loading 2D
[poses datasets](target-poses-and-bboxes-dataset), and visualising them
as points overlaid on video frames.

:::{warning}
This plugin is still in early stages of development and offers
limited functionality. We are working on ironing out the kinks and
gradually adding more features. [Get in touch](target-get-in-touch)
if you find any bugs or have suggestions for improvements!
:::

## Installation

The `napari` plugin is shipped with the `movement` package starting from
version `0.1.0`. If you install `movement` via `conda`, `napari` is already
included as a dependency. If you use the `pip` installer, make sure to
install `movement` with the `[napari]` extra:

```sh
pip install movement[napari]
```

## Launch the plugin

Type the following command in your terminal:

```sh
napari -w movement
```

This will open the `napari` window with the `movement` plugin docked on the
right-hand side.

## Load a background layer

Next, you need a background for you visualisation. You can either
[load the video](target-load-video) corresponding to the poses dataset,
or a [single image](target-load-frame), e.g., a still frame
derived from that video. In the following sections, we will show you how to
do both and discuss some limitations.

(target-load-video)=
### Load a whole video

To load a video, drag and drop the video file onto the `napari` window.
You will see a pop-up dialog asking you to select the reader.
Choose the `video` reader—corresponding to the
[`napari-video`](https://github.com/janclemenslab/napari-video)
plugin—and click `OK`.

`napari-video` will load the video as a single `napari`
[image layer](napari:howtos/layers/image.html), with a slider
at the bottom that you can use to navigate through frames.
You may also use the left and right arrow keys to navigate
frame-by-frame.

Clicking on the play button will start the video playback at a default
rate of 10 frames per second. You can adjust that by right-clicking on the
play button or by opening the `napari > Preferences` menu and changing
the `Playback frames per second` setting.

:::{admonition} Video playback limitations
:class: warning

- During playback you cannot jump at an arbitrary frame by clicking on the
  slider. Make sure to pause the video first.
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

You can extract a still frame using your video player/editor of
choice. We find the command line tool [`ffmpeg`](https://www.ffmpeg.org/)
very useful for this task.

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
In any case, the image will be loaded as a static
[image layer](napari:howtos/layers/image.html).

## Load the poses dataset

Now you are ready to load some pose tracks over your chosen background layer.

The `movement` plugin on the right-hand side of the window should contain
an expanded `Load poses` menu. This contains a `source software` dropdown,
an `fps`  (frames per second) input field, and a `Browse` button to select
a file containing predicted poses.
You may also directly paste a `file path` into the homonymous field.

::: {note}
See [supported formats](target-supported-formats) for more information on
the expected software and file formats.
:::

Once the file path is set, click the `Load` button to load the poses dataset
into `napari`. The plugin will create a new
[points layer](napari:howtos/layers/points.html) on top of the existing
image layer.

You will see a view similar to the one below:

![napari plugin with poses dataset loaded](../_static/napari_plugin_with_poses_as_points.png)

The predicted keypoints are represented as points, colour-coded by
keypoint ID for single-individual datasets, or by individual ID for
multi-individual datasets. Hovering with your mouse over a point will
bring up a tooltip containing the names of the individual and keypoint,
the point-wise confidence score (as predicted by the source software),
and the time in seconds (this is calculated from the frame number and
the `fps` value you provided).

Using the slider at the bottom of the window, you can navigate through
the frames of the poses dataset. The points will update accordingly, allowing
you to inspect the predicted keypoints at different time points.
If you loaded a video as a background layer, the video will also update
in sync with the poses dataset.

::: {admonition} Stay tuned
Though the display style of the points layer is currently fixed, we are
working on adding more customisation options in future releases, such as
enabling you to change the point size, colour, or shape.

We are also working on enabling the visualisation of
[bounding boxes datasets](target-poses-and-bboxes-dataset) in the plugin.
:::
