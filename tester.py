import cv2
import requests
import time

from helpers import parse_env, convert_np_im_to_base64, convert_base64_im_to_np

config = parse_env()
url = config['APPLICATION_EXTERNAL_URL']
port = config['APPLICATION_EXTERNAL_PORT']
api_method = 'v1.0/image/run-segmentation'

test_im = cv2.imread('test_im.jpg')
responce = requests.post(f'http://{url}:{port}/{api_method}', json={'im_base64': convert_np_im_to_base64(test_im)})
print(responce)
responce_im = convert_base64_im_to_np(responce.json()['im_base64'])
cv2.imwrite('responce.jpg', responce_im)