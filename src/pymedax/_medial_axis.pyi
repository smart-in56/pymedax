from typing import Literal, Tuple, Union, overload

import numpy as np
import numpy.typing as npt


Axis = npt.NDArray[np.uint8]
DistanceMap = npt.NDArray[np.float64]
ClosestBoundaryMap = npt.NDArray[np.int64]
FullResult = Tuple[Axis, DistanceMap, ClosestBoundaryMap]


@overload
def medial_axis(
    image: npt.NDArray[np.uint8],
    return_maps: Literal[False] = False,
    angle_threshold_degrees: float = 45.0,
    minimum_distance: float = 0.0
) -> Axis: ...


@overload
def medial_axis(
    image: npt.NDArray[np.uint8],
    return_maps: Literal[True],
    angle_threshold_degrees: float = 45.0,
    minimum_distance: float = 0.0
) -> FullResult: ...


@overload
def medial_axis(
    image: npt.NDArray[np.uint8],
    return_maps: bool,
    angle_threshold_degrees: float = 45.0,
    minimum_distance: float = 0.0
) -> Union[Axis, FullResult]: ...
