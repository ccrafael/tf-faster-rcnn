# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import threading
import matplotlib.pyplot as plt

import numpy as np
import PIL

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os

from utils.timer import Timer
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import bbox_transform_inv
from model.nms_wrapper import nms

from model import augment
from model import fusion

DEBUG = False

def _get_image_blob(image):
    """Converts an image into a network input.
    Arguments:
      image: a PIL Image.

    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """

    width, height = image.size
    im_size_min = min(width, height)
    im_size_max = max(width, height)

    processed_ims = []
    im_scale_factors = []

    # scale the image to net input
    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)

        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)

        image = image.resize((int(round(width*im_scale)), int(round(height*im_scale))), PIL.Image.BILINEAR)

    im_scale_factors.append(im_scale)

    # from PIL.Image to Numpy
    image = np.asarray(image, dtype=np.float32)

    # RGB to BGR
    image = np.flip(image, axis=2)
    image -= cfg.PIXEL_MEANS

    processed_ims.append(image)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_blobs(image):
    """Convert an image and RoIs within that image into network inputs."""

    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(image)

    return blobs, im_scale_factors


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""
    for i in range(boxes.shape[0]):
        boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

    return boxes


def im_detect(sess, net, image):
    """

    :param sess: Tensor flow session
    :param net: ConvNet
    :param im: A PIL Image
    :return: The scores and BB in a tuple with shapes (300x21), (300x84)
    """
    blobs, im_scales = _get_blobs(image)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']

    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

    #
    # Here is where the magic starts
    #
    _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])

    boxes = rois[:, 1:5] / im_scales[0]
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, np.array(image).shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue

            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]

            inds = np.where((x2 > x1) & (y2 > y1))[0]
            dets = dets[inds, :]
            if dets == []:
                continue

            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def test_net(tfsession, net, imdb, weights_filename, max_per_image=100, thresh=0.):
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

    threads = []
    predictions = {}

    # the function that is going to process the image and detect objects
    def worker(id, image, augmentation):
        # this isnot thread safe
        _t[id].tic()
        augmented_image = augmentation.augment(image)
        _t[id].toc()


        """
        augmented_image = augmented_image[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(augmented_image, aspect='equal')
        plt.show()
        """

        scores, boxes = im_detect(tfsession, net, augmented_image)

        _t[id].tic()
        boxes = augmentation.revert(image, augmented_image, boxes)
        _t[id].toc()

        # append is syncronized as far as I know
        predictions[id] = ([scores, boxes])

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
                     augment.Color(0.5),
                     augment.Scale(1, 0.9, 21),
                     augment.Scale(0.9, 1, 21),
                     augment.Rotation(90),
                     augment.Rotation(180),
                     augment.Rotation(270),)

    # init a timer for each augmentation
    for id in range(len(augmentations)):
        _t[id] = Timer()

    #ImageFile.LOAD_TRUNCATED_IMAGES = True

    for i in range(num_images):
    #for i in range(10):
        im = PIL.Image.open(imdb.image_path_at(i)).convert("RGB")

        _t['im_detect'].tic()

        predictions.clear()

        for j, augmentation in enumerate(augmentations):
            thread = threading.Thread(target=worker, args=(j, im, augmentation))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        _t['im_detect'].toc()
        _t['misc'].tic()

        # skip j = 0, because it's the background class
        detections = []

        for pred in predictions.values():
            scores, boxes = pred

            tmp = []
            for j, cls in enumerate(imdb.classes):
                if j == 0:
                    continue

                inds = np.where(scores[:, j] > thresh)[0]
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, j * 4:(j + 1) * 4]

                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = nms(cls_dets, cfg.TEST.NMS)

                cls_dets = cls_dets[keep, :]

                threshold_filter = np.where(cls_dets[:, 4] >= cfg.TEST.NMS)[0]

                cls_dets = cls_dets[threshold_filter, :]

                cls_dets = np.insert(cls_dets, 5, j, axis=1)
                tmp.append(cls_dets)

            detections.append(np.vstack(tmp))

        # all_boxes shape 21 x ~5000 x N x 6
        final_detections = fusion.fusion(detections)

        if len(final_detections) > 0:
            for j, cls in enumerate(imdb.classes):
                if j == 0:
                    continue
                inds = np.where(final_detections[:, 5] == j)[0]
                cls_dets = final_detections[inds, :-1]
                all_boxes[j][i] = cls_dets


        # just for debug
        if DEBUG:
            #au = augmentations[0]
            #im = au.augment(im)
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(im, aspect='equal')

            color = ['red', 'yellow', 'orange', 'purple', 'brown', 'blue', 'gray', 'pink', 'white', 'black', 'cyan']
            for j, pred in enumerate(predictions.values()):
                visualize_detections(imdb, pred[0], pred[1], color[j%len(color)], ax, thresh=cfg.TEST.NMS)

            visualize_labels(imdb, i, roi, ax)
            plt.show()

        _t['misc'].toc()

        aug_time = 0
        for j in range(len(augmentations)):
            aug_time += _t[j].average_time

        print('im_detect: {:d}/{:d} {:.3f}s misc: {:.3f}s augment: {:.3f}'
              .format(i + 1, num_images, _t['im_detect'].average_time, _t['misc'].average_time, aug_time/len(augmentations)))

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)


def visualize_labels(imdb, i, roi, ax):
    """ draw the labeled classes of the given image
    param i the index of the image on pascal voc 2007.
    """
    # roi is a list of dictionaries like:
    # {boxes: [[1,2,3,4], [1,2,3,4]], flipped: False,
    #    gt_classes: [L1, L2], gt_overlaps: (2x21 np matrix), set_areas: []}

    for l, bb in enumerate(roi[i]['boxes']):
        draw_bb(bb, imdb.classes[roi[i]['gt_classes'][l]], None,'green', ax, pos=3)


def visualize_detections(imdb, scores, boxes, color, ax, thresh=0.5):
    """Draw detected bounding boxes."""

    for j, cls in enumerate(imdb.classes):
        if j == 0:
            continue

        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]

        cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, cfg.TEST.NMS)

        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= thresh)[0]

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            draw_bb(bbox, cls, score, color, ax, pos=random.randint(0, 3))

    plt.axis('off')


def draw_bb(bbox, class_name, score, color, ax, pos=0):
    """ Draw a Boundig Box """

    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor=color, linewidth=1.5))

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
            fontsize=14, color='white')
