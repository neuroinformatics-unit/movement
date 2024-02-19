.. _target-api:

API Reference
=============


Input/Output
------------
.. currentmodule:: movement.io.load_poses
.. autosummary::
    :toctree: api

    from_file
    from_sleap_file
    from_dlc_file
    from_dlc_df
    from_lp_file

.. currentmodule:: movement.io.save_poses
.. autosummary::
    :toctree: api

    to_dlc_file
    to_dlc_df
    to_sleap_analysis_file
    to_lp_file

.. currentmodule:: movement.io.validators
.. autosummary::
    :toctree: api

    ValidFile
    ValidHDF5
    ValidPosesCSV
    ValidPoseTracks

Sample Data
-----------
.. currentmodule:: movement.sample_data
.. autosummary::
    :toctree: api

    list_sample_data
    fetch_sample_data_path
    fetch_sample_data

Logging
-------
.. currentmodule:: movement.logging
.. autosummary::
    :toctree: api

    configure_logging
    log_error
    log_warning
