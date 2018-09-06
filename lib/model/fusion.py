from sklearn.cluster import KMeans
import numpy as np
from model.nms_wrapper import nms
from model.config import cfg
import math


def kmeans(boxes, k):
    """
    Group into k clusters the BB in boxes.
    http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans


    :param boxes: The BB in format Nx4 where (x1,y1,x2,y2)
    :param k: the number of clusters.

    :return: k clusters with the element indexes of each clusters.
    """
    model = KMeans(n_clusters=k).fit(boxes)

    pred = model.labels_

    indexes = [[]] * k
    for i, v in enumerate(pred):
        indexes[v] = indexes[v] + [i]

    return indexes


def fusion(boxes):
    """
    Fusion the detections. All the detections belongs to a single class.

    :param boxes: A list of arrays of detections. The detections are in the form of (Nx6)
    and the size of the list of the main array correspond with the number of aumentation done
    before the detection. For each aumentation there is an array with shape Nx6. With format
    (x1,y1,x2,y2,score,label).

    The algorithm must group the detections into S object groups using k-means where k = S. And S is the
    largest number of detection in each variation.

    For each group fussion the BB using means, NMS, median or other simple method.

    :return: An array with shape Nx5 with the fusiones BB.
    """

    # if there is just one element dont do fusion
    if len(boxes) == 1:
        return boxes[0]

    S = max(map(lambda x: np.shape(x)[0], boxes))

    # stack it all together
    boxes = np.vstack(boxes)

    # if no elements return empty
    if len(boxes) == 0:
        return np.array([])

    print("Calculate  kmeans for %d boxes and k = %d"%( len(boxes), S))

    # group the BB, scores are ignored, must be S clusters
    clusters = kmeans(boxes[:, 0:4], S)

    new_boxes = []
    for i in range(S):
        new_boxes.append(fusion_median(boxes[clusters[i], :]))

    #print(np.vstack(new_boxes))
    return np.vstack(new_boxes)


def fusion_mean(boxes):
    labels_counter = np.zeros(21)
    for bb in boxes:
        labels_counter[int(bb[5])] += 1

    j = np.argmax(labels_counter)

    return np.array([np.mean(boxes[:, i]) for i in range(4)] + [np.mean(boxes[:, 4])] + [j])


def fusion_median(boxes):
    labels_counter = np.zeros(21)
    for bb in boxes:
        labels_counter[int(bb[5])] += 1

    j = np.argmax(labels_counter)

    return np.array([np.median(boxes[:, i]) for i in range(4)] + [np.median(boxes[:, 4])] + [j])


def fusion_nms(boxes):
    labels_counter = np.zeros(21)
    for bb in boxes:
        labels_counter[int(bb[5])] += 1

    j = np.argmax(labels_counter)

    keep = nms(boxes, cfg.TEST.NMS)

    boxes = np.insert(boxes[keep, :-1], 4, max(boxes[keep, 4]), axis=1)
    boxes = np.insert(boxes, 5, j, axis=1)
    inds = np.where(boxes[:, -1] >= cfg.TEST.NMS)[0]

    return boxes[inds, :]
