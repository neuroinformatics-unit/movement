"""
Export pose tracks to NWB
============================

Export pose tracks to NWB
"""

import datetime

from pynwb import NWBFile

from movement import sample_data
from movement.io.nwb_export import convert_movement_to_nwb

# Load the sample data
ds = sample_data.fetch_sample_data("DLC_two-mice.predictions.csv")

# The dataset has two individuals
# we will create two NWBFiles for each individual

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

# Convert the dataset to NWB
# This will create PoseEstimation and Skeleton objects for each individual
# and add them to the NWBFile
convert_movement_to_nwb(nwbfiles, ds)
