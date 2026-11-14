## Release [v0.18.0](https://github.com/neuroinformatics-unit/movement/releases/tag/v0.18.0) is out

This is a big release, jam-packed with lots of new features in our GUI, a more streamlined saving interface, and many new kinematic measures.

:warning: **This release contains minor breaking changes** to the saving functions, to the `nan_policy` options of path metrics, and to the default behaviour of `filter_by_confidence()`. See the [release notes](https://github.com/neuroinformatics-unit/movement/releases/tag/v0.18.0) for how to update your code.

To update movement to the latest version, see the [update guide](https://movement.neuroinformatics.dev/latest/user_guide/installation.html#update-the-package).

### :high_voltage: Highlight: Edit pose tracks in the GUI

You can now correct pose estimation errors by hand, directly in the `movement` GUI: move or remove keypoints, keep track of your edits on a timeline, and save the corrected dataset. See the [Edit tracked data](https://movement.neuroinformatics.dev/latest/user_guide/gui.html#edit-tracked-data) section of the GUI guide to get started.

This feature is the result of @**Anna Teruel**'s [Google Summer of Code 2026 project](https://neuroinformatics.dev/blog/gsoc_2026_movement_edit_widget.html). A huge thank you to Anna for designing and building it, and congratulations on joining the `movement` team! We're also grateful to @**Sofía Miñano**, who co-mentored the project and made the GUI's layer callbacks and frame slider more robust.

### :floppy_disk: Unified `save_dataset` entry point and other I/O improvements

Following the unified loading interface introduced in v0.14.0, @**Chang Huan Lo** has worked the same magic for saving: [`save_dataset()`](https://movement.neuroinformatics.dev/latest/api/movement.io.save.save_dataset.html) saves a `movement` dataset to a file in any supported format.

Thanks to @**Ishan Khan**, you can now also load pose tracks from [COCO keypoint results](https://movement.neuroinformatics.dev/latest/api/movement.io.load_poses.from_coco_file.html) files.

### :triangular_ruler: New kinematic measures: angular derivatives and path metrics

You can now compute the [angular velocity](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_angular_velocity.html) and [angular acceleration](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_angular_acceleration.html) of 2D orientations, such as head direction. See the new [example](https://movement.neuroinformatics.dev/latest/examples/compute_angular_derivatives.html) for a walkthrough.

Thanks to @**Sanjana** and @**Vybhav**, there are also two new path metrics: [sinuosity](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_path_sinuosity.html) and [maximum expected displacement](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_path_emax.html).

### :beaver: New example: annotating behavioural events with BORIS

Thanks to @**Holly Morley**, a new [example](https://movement.neuroinformatics.dev/latest/examples/advanced/annotate_boris_events.html) shows how to manually annotate events in [BORIS](https://www.boris.unito.it/) and load them into a `movement` dataset as per-frame labels.

### :handshake: Community and contributor experience

We have written down how the `movement` project is run: the new [governance page](https://movement.neuroinformatics.dev/latest/community/governance.html) describes the roles of contributors, team members and maintainers, and explains how decisions are made. We have also introduced pull request review limits for new contributors, described in the [contributing guide](https://movement.neuroinformatics.dev/latest/community/contributing.html).

See the [release notes](https://github.com/neuroinformatics-unit/movement/releases/tag/v0.18.0) for more info and a full list of changes.
