API Reference
=============


Input/Output
------------
.. currentmodule:: movement.io.load_poses
.. autosummary::
    :toctree: auto_api

    from_sleap_file
    from_dlc_file
    from_dlc_df

.. currentmodule:: movement.io.save_poses
.. autosummary::
    :toctree: auto_api

    to_dlc_file
    to_dlc_df

.. currentmodule:: movement.io.validators
.. autosummary::
    :toctree: auto_api

    ValidFile
    ValidHDF5
    ValidPosesCSV

Datasets
--------
.. currentmodule:: movement.datasets
.. autosummary::
    :toctree: auto_api

    find_pose_data
    fetch_pose_data_path

Logging
-------
.. currentmodule:: movement.logging
.. autosummary::
    :toctree: auto_api

    configure_logging
    log_and_raise_error
    log_warning
