"""Compute variables derived from ``position`` data.

Kinematics functions preserve unrelated input attributes, including ``log``
and custom metadata. Spatial outputs retain the input's ``units`` label,
and angles use ``radians`` or ``degrees``. Dimensionless outputs omit
``units``. Time derivatives and directional change derive rates only from
explicit numeric time metadata; sinuosity derives inverse square-root length
units only from supported simple labels. See ``compute_time_derivative``
for supported labels. No arbitrary unit expressions are parsed or values
converted. Kinetic energy omits ``units`` because mass units are unspecified.
Pairwise distances recognize length metrics and their SciPy aliases, and
derive squared units for squared Euclidean distances with supported labels.
All other uncertain output units are omitted. Attribute preservation does not
depend on the global xarray ``keep_attrs`` setting.
"""

from movement.kinematics.distances import compute_pairwise_distances
from movement.kinematics.kinematics import (
    compute_acceleration,
    compute_backward_displacement,
    compute_forward_displacement,
    compute_speed,
    compute_time_derivative,
    compute_velocity,
)
from movement.kinematics.kinetic_energy import compute_kinetic_energy
from movement.kinematics.orientation import (
    compute_forward_vector,
    compute_forward_vector_angle,
    compute_head_direction_vector,
)
from movement.kinematics.path import (
    compute_directional_change,
    compute_path_deviation,
    compute_path_emax,
    compute_path_length,
    compute_path_sinuosity,
    compute_path_straightness,
    compute_turning_angle,
)

__all__ = [
    "compute_forward_displacement",
    "compute_backward_displacement",
    "compute_velocity",
    "compute_acceleration",
    "compute_speed",
    "compute_path_length",
    "compute_path_straightness",
    "compute_directional_change",
    "compute_path_deviation",
    "compute_path_emax",
    "compute_time_derivative",
    "compute_path_sinuosity",
    "compute_pairwise_distances",
    "compute_forward_vector",
    "compute_head_direction_vector",
    "compute_forward_vector_angle",
    "compute_kinetic_energy",
    "compute_turning_angle",
]
