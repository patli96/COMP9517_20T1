import os
import time
import pickle
import argparse
import json

from bottle import run, post, request, response, route

import numpy as np
import cv2

from keras.layers import Input
from keras.models import Model
from keras_csp import config, bbox_process
from keras_csp.utilsfunc import *
from keras_csp import resnet50 as nn

# Get Args
parser = argparse.ArgumentParser(
    description='CSP Listener',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '-iw',
    '--image-width',
    help='Image width in pixels.',
    dest='width',
    type=int,
    default=768,
)
parser.add_argument(
    '-ih',
    '--image-height',
    help='Image height in pixels.',
    dest='height',
    type=int,
    default=576,
)
parser.add_argument(
    '--port',
    help='Port to be listened.',
    dest='port',
    type=int,
    default=8098,
)
parser.add_argument(
    '--weight',
    help='Path of the weight file',
    dest='weight',
    default='D:/DOWNLOAD/models_CSP/cityperson/withoffset/net_e121_l0.hdf5',
)
parser.add_argument(
    '--cpu',
    help='Run in CPU mode',
    dest='cpu',
    action='store_true',
)
parser.add_argument(
    '--gpu-id',
    help='Specify GPU ID',
    dest='gpu_id',
    default='0',
)
args = parser.parse_args()

C = config.Config()
C.gpu_ids = args.gpu_id
if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '65535'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = C.gpu_ids
C.onegpu = 4
C.offset = True
C.scale = 'h'
#C.size_test = (480, 640)
C.size_test = (args.height, args.width)
input_shape_img = (C.size_test[0], C.size_test[1], 3)

img_input = Input(shape=input_shape_img)

# define the network prediction
preds = nn.nn_p3p4p5(img_input, offset=C.offset, num_scale=C.num_scale, trainable=True)
model = Model(img_input, preds)

weight_path = args.weight
model.load_weights(weight_path, by_name=True)


@route('/ready')
def is_ready():
    return 'true'


@post('/process')
def csp_process():
    try:
        req = json.loads(request.body.read())
        score = req['score']
        img = np.array(req['img'], np.uint8)
        assert (img.shape[0], img.shape[1]) == C.size_test
    except (TypeError, AssertionError, IndexError):
        return 'null'

    x_rcnn = format_img(img, C)
    Y = model.predict(x_rcnn)

    if C.offset:
        boxes = bbox_process.parse_det_offset(Y, C, score=score, down=4)
    else:
        boxes = bbox_process.parse_det(Y, C, score=score, down=4, scale=C.scale)

    boxes[:, [0, 1, 2, 3, 4]] = boxes[:, [1, 0, 3, 2, 4]]
    return json.dumps(boxes.tolist())


run(host='localhost', port=args.port, debug=True)
