import time

import os
import glob
import cv2

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.cluster import KMeans

from PIL import Image
from skimage import filters

import redis
from flask import Flask, request, jsonify

from helpers import convert_base64_im_to_np, convert_np_im_to_base64

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)

def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)

@app.route('/')
def hello():
    count = get_hit_count()
    return 'Hello World! I have been seen {} times.\n'.format(count)

def k_means_segmentation(source_image, color_scheme='mean', n_clusters=7):
    """
    Takes picture to apply K-Means segmentation. 
    Returns a tuple with images, processed in RGB.
    
    Argruments:
    -------------
    source_image, np.array with shape (x, y, 3):
        Image to segmentate.
    color_scheme, str:
        Color scheme that will be used. 
        Possible variants are 'mean' and 'experimental'.
        'mean' - there'll be computed and assigned mean colors for each cluster.
        'experimental' - beautiful 'deep orange and purple ' color palette will be used.
        
        'mean' is set by default.
    n_clusters, int:
        Number of cluster to apply segmentation.
        
        7 is set by default.
    """
    df = pd.DataFrame(columns=['x', 'y', 'R', 'G', 'B'])

    im = source_image.copy()

    pixels_data = []

    for i in range(len(im)):
        for j in range(len(im[i])):
            rgb = im[i][j]

            new_pixel = {}
            new_pixel['x'] = i
            new_pixel['y'] = j
            new_pixel['R'] = rgb[0]
            new_pixel['G'] = rgb[1]
            new_pixel['B'] = rgb[2]

            pixels_data.append(new_pixel)

    df = df.append(pixels_data, ignore_index=True)

    kmeans = KMeans(n_clusters=n_clusters)
    df['cluster_rgb'] = kmeans.fit_predict(df[['R', 'G', 'B']])

    channels_list = ['R', 'G', 'B']
    
    if color_scheme  == 'mean':
        mean_cluster_values_rgb = {i : {ch: np.mean(df[df['cluster_rgb'] == i][ch]) for ch in channels_list}
                                    for i in range(n_clusters)}

        for index, row in df.iterrows():
            for ch in channels_list:
                df.at[index, ch] = mean_cluster_values_rgb[row['cluster_rgb']][ch]
    elif color_scheme == 'experimental':
        # try it!
        mean_cluster_values_rgb = {i : {ch: np.mean(df[df['cluster_rgb'] == i][ch]) for ch in channels_list}
                                    for i in range(n_clusters)}
        
        mean_cl_colors_brigthnesses = [[i, (0.3*mean_cluster_values_rgb[i]['R'] + 
                                            0.59*mean_cluster_values_rgb[i]['G'] + 
                                            0.11*mean_cluster_values_rgb[i]['B'])]
                                       for i in range(n_clusters)]
        mean_cl_colors_brigthnesses = sorted(mean_cl_colors_brigthnesses, key = lambda color: color[1])
        # assigning brightness rank 
        for i in range(n_clusters):
            mean_cl_colors_brigthnesses[i].append(i)
        # going back to cluster numbers
        mean_cl_colors_brigthnesses = sorted(mean_cl_colors_brigthnesses, key = lambda color: color[0])
        
        clusters_color_channels = {}
        for i in range(n_clusters):
            curr_color_point = mean_cl_colors_brigthnesses[i][2]*len(red.values())/n_clusters
            clusters_color_channels[i] = {}
            clusters_color_channels[i]['R'] = np.interp(xp=list(red.keys()), fp=list(red.values()), x=curr_color_point)
            clusters_color_channels[i]['G'] = np.interp(xp=list(green.keys()), fp=list(green.values()), x=curr_color_point)
            clusters_color_channels[i]['B'] = np.interp(xp=list(blue.keys()), fp=list(blue.values()), x=curr_color_point)
        for index, row in df.iterrows():
            for ch in channels_list:
                df.at[index, ch] = clusters_color_channels[row['cluster_rgb']][ch]
                
    # assigning results we've got
    
    w = len(im)
    h = len(im[0])
    
    for i in range(w):
        for j in range(h):
            im[i][j][0] = df.iloc[h*i + j]['R']
            im[i][j][1] = df.iloc[h*i + j]['G']
            im[i][j][2] = df.iloc[h*i + j]['B']           
            
    return im


@app.route('/v1.0/image/run-segmentation', methods=['POST'])
def run_segmentation():
    if not request.json or not 'im_base64' in request.json:
        return jsonify({'error_msg': 'No image provided'}, 400)
    request_json = request.json # request_json is a dict, containing JSON, received in HTTP POST request
    im_base64 = request_json['im_base64'] # getting image from request JSON
    im_np = convert_base64_im_to_np(im_base64)

    # DO SOME MANIPULATIONS HERE
    im_np = k_means_segmentation(im_np, n_clusters=int(request_json['n_segments']))

    transformed_image = convert_np_im_to_base64(im_np) # put transformed image in here
    return jsonify({'im_base64': transformed_image}), 200
