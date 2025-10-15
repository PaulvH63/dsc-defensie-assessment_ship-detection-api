import numpy as np
import pandas as pd
from typing import Tuple

def rle_decode(mask_rle: str, shape: Tuple[int, int] = (768, 768)) -> np.ndarray:
    """Decode a run-length encoded (RLE) mask into a 2D numpy array.

    Args:
        mask_rle: A run-length encoded string (e.g. "1 3 10 5 ...").
                  If None or NaN, an empty mask is returned.
        shape: The (height, width) of the output mask.

    Returns:
        A 2D numpy array of shape `shape`, where 1 represents
        the mask and 0 the background.
    """
    if mask_rle is None or pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)

    rle_numbers = list(map(int, mask_rle.split()))
    starts = np.array(rle_numbers[0::2], dtype=int) - 1
    lengths = np.array(rle_numbers[1::2], dtype=int)
    ends = starts + lengths

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1

    return mask.reshape(shape).T


def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary mask into a run-length encoded (RLE) string.

    The mask is encoded in column-major order (Fortran order), consistent
    with Kaggle's RLE format. Runs are defined by transitions from 0 to 1
    and 1 to 0.

    Args:
        mask: A 2D numpy array of shape (H, W) with values 0 or 1.

    Returns:
        A run-length encoded string (e.g. "1 3 10 5 ...").
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    starts = runs[::2]
    lengths = runs[1::2] - runs[::2]
    return ' '.join(str(x) for pair in zip(starts, lengths) for x in pair)

