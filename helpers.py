import io
import base64
import numpy as np

from PIL import Image

def parse_env(env_file:str = '.env') -> dict:
    """
    Creates dict with contents of .env file
    """
    with open(env_file, 'r') as f:
        contents = f.readlines()
    result = dict()
    for line in contents:
        if '=' in line:
            key, value = line.replace(' ', '').replace('\n', '').split('=')
            result[key] = value
    return result


def convert_base64_im_to_np(base64_im: str) -> np.ndarray:
    """
    Converts base64 encoded string, representing image to numpy array
    """
    decoded_im = base64.b64decode(base64_im)
    pil_im = Image.open(io.BytesIO(decoded_im))
    np_im = np.array(pil_im)
    return np_im

def convert_np_im_to_base64(np_im: np.ndarray, format: str = 'JPEG') -> str:
    """
    Converts numpy array, representing image to base64 encoded string
    """
    pil_im = Image.fromarray(np_im)
    buff = io.BytesIO()
    pil_im.save(buff, format="JPEG")
    base64_im = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64_im
