.. _target-api:

API Reference
=============


Input/Output
------------
.. currentmodule:: movement.io.load_poses
.. autosummary::
    :toctree: api

    from_numpy
    from_file
    from_sleap_file
    from_dlc_file
    from_lp_file
    from_dlc_style_df

.. currentmodule:: movement.io.save_poses
.. autosummary::
    :toctree: api

    to_dlc_file
    to_lp_file
    to_sleap_analysis_file
    to_dlc_style_df


.. currentmodule:: movement.io.validators
.. autosummary::
    :toctree: api

    ValidFile
    ValidHDF5
    ValidDeepLabCutCSV
    ValidPosesDataset

Sample Data
-----------
.. currentmodule:: movement.sample_data
.. autosummary::
    :toctree: api

    list_datasets
    fetch_dataset_paths
    fetch_dataset

Filtering
---------
.. currentmodule:: movement.filtering
.. autosummary::
    :toctree: api

    filter_by_confidence
    median_filter
    savgol_filter
    interpolate_over_time
    report_nan_values


Analysis
-----------
.. currentmodule:: movement.analysis.kinematics
.. autosummary::
    :toctree: api

    compute_displacement
    compute_velocity
    compute_acceleration

.. currentmodule:: movement.utils.vector
.. autosummary::
    :toctree: api

    cart2pol
    pol2cart

MovementDataset
---------------
.. currentmodule:: movement.move_accessor
.. autosummary::
    :toctree: api

    MovementDataset

Logging
-------
.. currentmodule:: movement.logging
.. autosummary::
    :toctree: api

    configure_logging
    log_error
    log_warning
