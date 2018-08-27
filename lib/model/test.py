# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Lock

import matplotlib.pyplot as plt

import numpy as np
import PIL

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os

from utils.timer import Timer

from model.config import cfg, get_output_dir

from model import augment
from model import fusion

DEBUG = False


def worker(args_list):
    id = args_list[0]
    image = args_list[1]
    augmentation = args_list[2]
    model = args_list[3]
    lock = args_list[4]

    augmented_image = augmentation.augment(image)

    # models doesnt like concurrence
    lock.acquire()
    pred = model.predict(augmented_image)
    lock.release()

    pred = augmentation.revert(image, augmented_image, pred)

    # append is syncronized as far as I know
    return {'id': id, 'prediction': pred}


def test_net(model, imdb, weights_filename):
    """Test a Fast R-CNN network on an image database.

    Keyword arguments:
    tfsession -- the tensorflow session
    net -- the network architecture to test
    imdb -- the class with the dataset of images to use as test set
    weights -- a path to the the already calculated weights, must be a tf format
    max_per_image --
    thresh --

    """

    np.random.seed(cfg.RNG_SEED)
    num_images = len(imdb.image_index)

    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(imdb.num_classes)]

    # just for debug, load the labels of the images
    if DEBUG:
        roi = imdb.gt_roidb()
        num_images = 10

    #
    output_dir = get_output_dir(imdb, weights_filename)

    # timers
    _t = {'misc': Timer(), 'im_detect': Timer()}

    # the function that is going to process the image and detect objects

    augmentations = (augment.Ident(),
                     augment.Brightness(0.25),
                     augment.Brightness(0.5),
                     augment.Brightness(1.5),
                     augment.Brightness(2),
                     augment.Brightness(2.5),
                     augment.Contrast(0.25),
                     augment.Contrast(0.5),
                     augment.Contrast(1.5),
                     augment.Contrast(2),
                     augment.Contrast(2.5),
                     augment.Equalization(),

                     augment.Sharpeness(0.5),
                     augment.Sharpeness(2),

                     augment.Color(0.5),
                     augment.Color(1.5),


                     augment.Scale(1, 0.8),
                     augment.Scale(0.8, 1),

                     augment.Rotation(90),
                     augment.Rotation(180),
                     augment.Rotation(270),)


    thread_pool = ThreadPool(len(augmentations))

    # init a timer for each augmentation
    for id in range(len(augmentations)):
        _t[id] = Timer()

    # ImageFile.LOAD_TRUNCATED_IMAGES = True

    lock = Lock()

    for i in range(num_images):
        im = PIL.Image.open(imdb.image_path_at(i)).convert("RGB")

        _t['im_detect'].tic()

        # prepare the arguments for the workers
        args = []
        for j, augmentation in enumerate(augmentations):
            args += [[j, im, augmentation, model, lock]]

        # load the results form the threads
        predictions_queue = thread_pool.map(worker, args)

        predictions = []
        for p in predictions_queue:
            predictions.append(p)

        _t['im_detect'].toc()
        _t['misc'].tic()

        # skip j = 0, because it's the background class
        detections = []

        # put all detections together
        for p in predictions:
            if len(p['prediction']) > 0:
                detections.append(p['prediction'])

        # just for debug what the hell is happening
        if DEBUG:
            #au = augmentations[0]
            #im = au.augment(im)
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(im, aspect='equal')

            color = ['red', 'yellow', 'orange', 'purple', 'brown', 'blue', 'gray', 'pink', 'white', 'black', 'cyan']
            for p in predictions:
                visualize_detections(imdb, p['prediction'], color[p['id'] % len(color)], ax)
                pass

            visualize_labels(imdb, i, roi, ax)
            plt.show()

        # if there no prediction stop here
        if len(detections) == 0:
            continue

        # all_boxes shape 21 x ~5000 x N x 6
        final_detections = fusion.fusion(detections)

        if len(final_detections) > 0:
            for j, cls in enumerate(imdb.classes):
                if j == 0:
                    continue
                inds = np.where(final_detections[:, 5] == j)[0]
                cls_dets = final_detections[inds, :-1]
                all_boxes[j][i] = cls_dets

        _t['misc'].toc()

        aug_time = 0
        for j in range(len(augmentations)):
            aug_time += _t[j].average_time

        print('im_detect: {:d}/{:d} {:.3f}s misc: {:.3f}s augment: {:.3f}'
              .format(i + 1, num_images, _t['im_detect'].average_time, _t['misc'].average_time,
                      aug_time / len(augmentations)))

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

    thread_pool.close()


def visualize_labels(imdb, i, roi, ax):
    """ draw the labeled classes of the given image
    param i the index of the image on pascal voc 2007.
    """
    # roi is a list of dictionaries like:
    # {boxes: [[1,2,3,4], [1,2,3,4]], flipped: False,
    #    gt_classes: [L1, L2], gt_overlaps: (2x21 np matrix), set_areas: []}

    for l, bb in enumerate(roi[i]['boxes']):
        draw_bb(bb, imdb.classes[roi[i]['gt_classes'][l]], None, 'green', ax, pos=3)


def visualize_detections(imdb, predictions, color, ax):
    """Draw detected bounding boxes."""

    for p in predictions:
        bbox = p[:4]
        score = p[4]
        cls = imdb.classes[int(p[5])]
        draw_bb(bbox, cls, score, color, ax, pos=1)

    plt.axis('off')


def draw_bb(bbox, class_name, score, color, ax, pos=0):
    """ Draw a Boundig Box """

    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor=color, linewidth=3))

    str_format = '{:s} {:.3f}'
    if score is None:
        str_format = '{:s}'

    x = int(bbox[0])
    y = int(bbox[1])

    if pos == 1:
        x = bbox[2]
        y = bbox[1]
    elif pos == 2:
        x = bbox[0]
        y = bbox[3]
    elif pos == 3:
        x = bbox[2]
        y = bbox[3]

    ax.text(x, y, str_format.format(class_name, score),
            bbox=dict(facecolor=color, alpha=0.5),
            fontsize=16, color='white')
