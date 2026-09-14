# pymedax

pymedax computes a medial axis, with the possibility of relaxing the classical medial-axis definition by requiring the fronts to originate from sufficiently different directions.

Adding this criterion can be useful in domains such as control and automation, where we may want to define a policy based on points at which a choice between significantly different directions must be made.

In this context, we are often interested in identifying points where, from one direction, we can reach one part of the boundary, while from another direction, we can reach a substantially different part of the boundary. This can be more useful than simply detecting points that have two different closest boundary points when those boundary points are located in roughly the same direction.

The implementation is based on a fire-propagation analogy. We simulate the propagation of fronts starting from the boundary. When two fronts meet, we create a medial-axis point only if the directions associated with the two fronts differ significantly according to a user-defined criterion.

This allows the definition of different medial-axis policies depending on the application. The classical medial axis can be seen as one particular policy, while more restrictive definitions can be used to identify only meaningful decision points where the available paths lead toward substantially different regions of the boundary.
## Install

```bash
python -m pip install -e ".[test]"
```

## Usage

```python
import numpy as np
from pymedax import medial_axis

image = np.zeros((32, 32), dtype=np.uint8)
image[6:26, 10:22] = 1
axis = medial_axis(image)

axis, distance, closest_boundary = medial_axis(
	image,
	return_maps=True,
	angle_threshold_degrees=5.0,
	minimum_distance=0.0,
)
```

The input must be a 2D, C-contiguous NumPy array with dtype `uint8`. Nonzero
pixels are foreground. The default 8-connected neighborhood is used.

With `return_maps=True`, the result is:

- `axis`: a `uint8` array with `1` at medial-axis pixels;
- `distance`: a `float64` Euclidean distance map (`inf` for background);
- `closest_boundary`: an `int64` flattened boundary-pixel index map (`-1` for background).

Boundary pixels are initialized as distance-zero sources. A pixel is marked on
the axis when a processed neighbor has a different closest-boundary source and
the largest angle between any two distinct arrived boundary sources is at least
`angle_threshold_degrees`. Unprocessed neighbors are ignored because their
front assignment is not known yet. Angles below the threshold are rejected.
The default threshold is 45 degrees. Equal-distance candidates keep the first
state encountered by the deterministic row-major source order and heap ordering.
Candidates whose Euclidean distance to their assigned boundary source is below
`minimum_distance` are not marked as medial-axis pixels. The default minimum
distance is 0 pixels.


## Examples

Here is the result of computing the medial axis on the image above, using the following parameters:

minimum_distance = 5.0: points with a distance smaller than 5.0 are ignored.
angle_threshold_degrees = 90: only fronts originating from directions differing by at least 90° are considered.

![Binary input](https://raw.githubusercontent.com/smart-in56/pymedax/v0.1.0/examples/data/america_binary.png)

![Medial axis result](https://raw.githubusercontent.com/smart-in56/pymedax/v0.1.0/examples/data/america_overlay.png)