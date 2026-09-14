import pymedax
import numpy as np
from PIL import Image
from pathlib import Path

import numpy as np
from scipy import ndimage

# read america binary image
binary = Image.open(Path(__file__).parent.joinpath("data", "america_binary.png"))
binary = np.array(binary).astype(np.uint8)

# fill holes in the binary image
filled = ndimage.binary_fill_holes(binary).astype(np.uint8)

# compute the medial axis of the filled image
axis = pymedax.medial_axis(filled, return_maps=False, angle_threshold_degrees=90.0, minimum_distance=5.0)

# save the medial axis as an image
Image.fromarray(axis * 255).save(Path(__file__).parent.joinpath("data", "america_axis.png"))

# overlay the medial axis on the filled image
overlay = np.array(filled * 255)  # convert filled binary image to RGB-like format
overlay = np.stack([overlay, overlay, overlay], axis=-1)  # make it 3-channel
overlay[axis != 0] = [255, 0, 0]  # mark the axis in red
Image.fromarray(overlay).save(Path(__file__).parent.joinpath("data", "america_overlay.png"))