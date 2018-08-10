import numpy as np
import PIL, ImageOps
from PIL import Image, ImageEnhance
import abc
import cv2


class Augment:
    """ Represent an augmentation to be applied to a image. """

    def __init__(self):
        pass

    @abc.abstractmethod
    def augment(self, image):
        """ Augment the image and return the augmented image.

        param image: A cv2 image.
        return A cv2 image.
        """
        pass

    def revert(self, image, augmented_image, boxes):
        """
        Revert the augmentation to the detected BB

        The augmentations that perform geometric transformations need to invert the augmentation on the BB
        detected.
        """
        return boxes


class Ident(Augment):
    """ A aumentation that do nothing. """

    def __init__(self):
        pass

    def augment(self, image):
        return image


class Contrast(Augment):
    """ Contrast aumentation. """

    def __init__(self, contrast):
        self._contrast = contrast

    def augment(self, image):
        """ Change the contrast of a image """

        contrast = ImageEnhance.Contrast(image)
        return contrast.enhance(self._contrast)


class Brightness(Augment):
    """ Brightness aumentation. """

    def __init__(self, value):
        self._value = value

    def augment(self, image):
        bright = ImageEnhance.Brightness(image)

        return bright.enhance(self._value)


class Color(Augment):
    """ Color aumentation. """

    def __init__(self, value):
        self._value = value

    def augment(self, image):
        color = ImageEnhance.Color(image)

        return color.enhance(self._value)


class Sharpeness(Augment):
    """ Sharpeness aumentation. """

    def __init__(self, value):
        self._value = value

    def augment(self, image):
        sharpeness = ImageEnhance.Sharpness(image)

        return sharpeness.enhance(self._value)


class Scale(Augment):
    """ scale augmentation."""

    def __init__(self, scalex, scaley):
        self._scalex = scalex
        self._scaley = scaley
        self._revert = [self._scalex, self._scaley, self._scalex, self._scaley]

    def augment(self, image):
        """ Change the scale """
        width, height = image.size
        return image.resize((int(round(width*self._scalex)), int(round(height* self._scaley))), PIL.Image.LANCZOS)

    def revert(self, image, image_augmented, predictedbb):

        if len(predictedbb) > 0:
            predictedbb[:, 0:4] = np.divide(predictedbb[:, 0:4], self._revert)
        return predictedbb


class Rotation(Augment):
    """ rotation augmentation
        Bounding box rotation
        https: // cristianpb.github.io / blog / image - rotation - opencv

    """

    def __init__(self, angle):
        self._angle = angle

    def augment(self, image):
        """ rotation augmentation """

        return self.rotate_bound(image, self._angle)

    def rotate_bound(self, image, angle):

        return image.rotate(angle, expand=True)

    def compute_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # centre
        (w, h) = image.size
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        return  nH, nW

    def revert(self, image, augmented_image, boundingBoxes):
        """ Revert the rotation for the boundig boxes """

        cols, rows = augmented_image.size
        ori_cols, ori_rows = image.size

        cx = cols//2.0
        cy = rows//2.0

        # calculate the new bounds after the revert
        nH, nW = self.compute_bound(augmented_image, -self._angle)

        # calculate the coordinates change because the rotations
        delta_width = (nW - ori_cols)//2
        delta_height = (nH - ori_rows)//2

        new_boxes = []
        for bb in boundingBoxes:

            # get a bounding box
            new_bb = [(bb[0], bb[1]), (bb[2], bb[1]), (bb[0], bb[3]), (bb[2], bb[3])]

            # revert the rotation of the BB
            new_bb = self.rotate_box(new_bb, cx, cy, rows, cols)

            # revert the offset of the BB
            new_bb = [(p[0] - delta_width, p[1] - delta_height) for p in new_bb]

            # take the BB of the BB
            new_bb = [max(0, min([x[0] for x in new_bb])),
                 max(0, min([x[1] for x in new_bb])),
                 min(image.size[0], max([x[0] for x in new_bb])),
                 min(image.size[1], max([x[1] for x in new_bb])), bb[4], bb[5]]

            new_boxes.append(new_bb)

        return np.array(new_boxes)

    def rotate_box(self, bb, cx, cy, h, w):

        new_bb = list(bb)
        for i, coord in enumerate(bb):
            # opencv calculates standard transformation matrix
            M = cv2.getRotationMatrix2D((cx, cy), -self._angle, 1.0)

            # Grab  the rotation components of the matrix)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - cx
            M[1, 2] += (nH / 2) - cy

            # Prepare the vector to be transformed
            v = [coord[0], coord[1], 1]

            # Perform the actual rotation and return the image
            calculated = np.dot(M, v)
            new_bb[i] = (calculated[0], calculated[1])

        return new_bb


class Equalization(Augment):
    """ Image equalization
    https://www.packtpub.com/mapt/book/application_development/9781785283932/2/ch02lvl1sec26/enhancing-the-contrast-in-an-image
    """

    def __init__(self):
        pass

    def augment(self, image):
        return ImageOps.equalize(image)