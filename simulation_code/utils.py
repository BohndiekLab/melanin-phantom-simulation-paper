from PIL import Image
import numpy as np


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img


def half_scale(image):
    new_image = np.zeros(tuple(x // 2 for x in image.shape))
    new_image += image[::2, ::2, ::2]
    new_image += image[1::2, ::2, ::2]
    new_image += image[::2, 1::2, ::2]
    new_image += image[::2, ::2, 1::2]
    new_image += image[1::2, 1::2, ::2]
    new_image += image[1::2, ::2, 1::2]
    new_image += image[::2, 1::2, 1::2]
    new_image += image[1::2, 1::2, 1::2]
    return new_image
