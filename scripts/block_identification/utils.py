import numpy as np
from PIL import Image

IMG_SIZE = 64  # resize POVs to this before comparing


def flatten_pov(pov: np.ndarray) -> np.ndarray:
    pov = np.array(pov, dtype=np.uint8)
    img = Image.fromarray(pov).resize((IMG_SIZE, IMG_SIZE))
    result = np.array(img).flatten().astype(np.float32) / 255.0
    return result