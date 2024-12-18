"""Interfacing with poses stored in NWB files
=============================================

Save pose tracks to NWB files and load them back into ``movement``.
"""

# %% Load the sample data
import datetime

import xarray as xr
from pynwb import NWBHDF5IO, NWBFile

from movement import sample_data
from movement.io.nwb import (
    ds_from_nwb_file,
    ds_to_nwb,
)

ds = sample_data.fetch_dataset("DLC_two-mice.predictions.csv")

# %%The dataset has two individuals.
# We will create two NWBFiles for each individual

session_start_time = datetime.datetime.now(datetime.timezone.utc)
nwbfile_individual1 = NWBFile(
    session_description="session_description",
    identifier="individual1",
    session_start_time=session_start_time,
)
nwbfile_individual2 = NWBFile(
    session_description="session_description",
    identifier="individual2",
    session_start_time=session_start_time,
)

nwbfiles = [nwbfile_individual1, nwbfile_individual2]

# %% Convert the dataset to NWB
# This will create PoseEstimation and Skeleton objects for each
# individual and add them to the NWBFile
ds_to_nwb(ds, nwbfiles)

# %% Save the NWBFiles
for file in nwbfiles:
    with NWBHDF5IO(f"{file.identifier}.nwb", "w") as io:
        io.write(file)

# %% Convert the NWBFiles back to a movement dataset
# This will create a movement dataset with the same data as
# the original dataset from the NWBFiles

# Convert each NWB file to a single-individual movement dataset
datasets = [
    ds_from_nwb_file(f) for f in ["individual1.nwb", "individual2.nwb"]
]

# Combine into a multi-individual dataset
ds = xr.merge(datasets)