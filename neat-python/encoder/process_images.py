# coding: utf-8

import gym
import numpy as np
from keras.models import model_from_json
import json
from Object_detection_features_new import *
from skimage.transform import resize
from skimage.color import rgb2gray

from keras import backend as K
K.set_image_dim_ordering('th')


'''

env_name = 'Skiing-v0'
env = gym.make(env_name)
max_observations = 100
images = []
aux_observations = []
render = False
count = 0

while True:
    if len(images) >= max_observations: break
    s = env.reset()
    done = False
    
    while not done:
        if render: env.render()
        if len(images) >= max_observations: break
        a = env.action_space.sample()
        s, r, done, _ = env.step(a)
        
        if count % 10 < 3:
            aux_observations.append(env.ale.getScreenGrayscale()[:, :, 0])
        elif count % 10 == 3:
            images.append(aux_observations)
            aux_observations = []
            if not len(images) % 500:
                print(len(images))
        a = env.action_space.sample()
        count += 1

env.close()
images = np.array(images)
'''


def process_images(env_name, images, verbose=True):
    output_shape = (60, 60)
    game_name = env_name.split('-')[0]

    if verbose: print('Initializing')
    env = gym.make(env_name)
    odf = ObjectDetectionFeatures2(env)

    if verbose: print('Simplifying and resizing')
    if images.ndim == 4:
        processed_images = np.empty(images.shape[:2] + output_shape, dtype='uint8')
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                processed_images[i, j] = (resize(odf.get_simple_image(images[i, j]), 
                                                 output_shape, order=0) * 255).astype('uint8')
    else:
        print('Please, provide grayscale images with shape = (N, 3, height, width)')
        return

    if verbose: print('Encoding images')
    with open('./data/{}_Encoder_08_02.txt'.format(game_name), 'r') as model_file:
         model = model_from_json(json.loads(next(model_file)))

    model.load_weights('./data/{}_Encoder_08_02.h5'.format(game_name))
    return model.predict(processed_images)


'''
env_name = 'Skiing-v0'
tmp = process_images(env_name, images)
'''