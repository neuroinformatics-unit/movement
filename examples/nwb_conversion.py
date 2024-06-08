"""Converting movement dataset to NWB or loading from NWB to movement dataset.
============================

Export pose tracks to NWB
"""

# %% Load the sample data
import datetime

from pynwb import NWBHDF5IO, NWBFile

from movement import sample_data
from movement.io.nwb import (
    add_movement_dataset_to_nwb,
    convert_nwb_to_movement,
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
add_movement_dataset_to_nwb(nwbfiles, ds)

# %% Save the NWBFiles
for file in nwbfiles:
    with NWBHDF5IO(f"{file.identifier}.nwb", "w") as io:
        io.write(file)

# %% Convert the NWBFiles back to a movement dataset
# This will create a movement dataset with the same data as
# the original dataset from the NWBFiles

# Convert the NWBFiles to a movement dataset
ds_from_nwb = convert_nwb_to_movement(
    nwb_filepaths=["individual1.nwb", "individual2.nwb"]
)
