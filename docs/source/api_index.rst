.. _target-api:

API Reference
=============


Input/Output
------------
.. currentmodule:: movement.io.load_poses
.. autosummary::
    :toctree: api

    from_sleap_file
    from_dlc_file
    from_dlc_df

.. currentmodule:: movement.io.save_poses
.. autosummary::
    :toctree: api

    to_dlc_file
    to_dlc_df

.. currentmodule:: movement.io.validators
.. autosummary::
    :toctree: api

    ValidFile
    ValidHDF5
    ValidPosesCSV
    ValidPoseTracks

Datasets
--------
.. currentmodule:: movement.datasets
.. autosummary::
    :toctree: api

    list_pose_data
    fetch_pose_data_path

Logging
-------
.. currentmodule:: movement.logging
.. autosummary::
    :toctree: api

    configure_logging
    log_error
    log_warning
