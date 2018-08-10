from model.bbox_transform import bbox_transform_inv
from utils.blob import im_list_to_blob
from Model import Model
import numpy as np
import PIL
from model.config import cfg
from model.nms_wrapper import nms


class FasterRCNN(Model):
    """
    The Faster-RCNN model.
    """

    def __init__(self, tfsession, net, imdb, max_per_image=100, thresh=0.3):
        super(FasterRCNN, self).__init__(tfsession, imdb)
        self.net = net
        self.max_per_image = 100
        self.thresh = thresh

    def im_detect(self, sess, net, image):
        """
        :param sess: Tensor flow session
        :param net: ConvNet
        :param im: A PIL Image
        :return: The scores and BB in a tuple with shapes (300x21), (300x84)
        """
        blobs, im_scales = self._get_blobs(image)
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
            pred_boxes = FasterRCNN._clip_boxes(pred_boxes, np.array(image).shape)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        tmp = []
        for j, cls in enumerate(self.imdb.classes):
            if j == 0:
                continue

            inds = np.where(scores[:, j] > self.thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = pred_boxes[inds, j * 4:(j + 1) * 4]

            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)

            cls_dets = cls_dets[keep, :]

            cls_dets = np.insert(cls_dets, 5, j, axis=1)

            threshold_filter = np.where(cls_dets[:, 4] >= cfg.TEST.NMS)[0]

            tmp.append(cls_dets[threshold_filter, :])

        return np.vstack(tmp)

    def predict(self, image):
        """ Predict the BB of a image """
        return self.im_detect(self.tfsession, self.net, image)

    @staticmethod
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

            image = image.resize((int(round(width * im_scale)), int(round(height * im_scale))), PIL.Image.BILINEAR)

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

    @staticmethod
    def _get_blobs(image):
        """Convert an image and RoIs within that image into network inputs."""

        blobs = {}
        blobs['data'], im_scale_factors = FasterRCNN._get_image_blob(image)

        return blobs, im_scale_factors

    @staticmethod
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

    @staticmethod
    def _rescale_boxes(boxes, inds, scales):
        """Rescale boxes according to image rescaling."""
        for i in range(boxes.shape[0]):
            boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

        return boxes

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
