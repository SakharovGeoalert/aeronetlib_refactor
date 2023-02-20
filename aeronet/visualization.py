import numpy as np


def add_mask_single(image: np.ndarray,
                    mask: np.ndarray,
                    color: tuple = (255, 0, 0),
                    intensity: float = 0.5):

    mask = mask.squeeze() / np.max(mask)
    mask = np.stack((color[0]*mask, color[1]*mask, color[2]*mask), axis=-1)*intensity
    image = np.clip(image.astype(np.uint16) + mask.astype(np.uint16), 0, 255).astype(np.uint8)
    return image


def add_mask(image: np.ndarray,
             mask: np.ndarray,
             colormap: tuple = ((255, 0, 0), (0, 255, 0), (0, 0, 255)),
             intensity: float = 0.5):
    for ch in range(mask.shape[2]):
        image = add_mask_single(image, mask[:, :, ch], colormap[ch], intensity)
    return image
