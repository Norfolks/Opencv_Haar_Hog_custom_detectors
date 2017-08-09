import numpy as np
import re


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    result = boxes[pick].astype("int")
    # result[2] = result[2] - result[0]
    # result[3] = result[3] - res
    return result


def get_list_of_annotation_data(ann_path):
    with open(ann_path, 'r', encoding='ISO-8859-15') as annotation_description:
        counter = 0
        coordinates = list()
        for line in annotation_description:
            if line.startswith('Bounding'):
                x_min, y_min, x_max, y_max = [int(el) for el in re.findall(r'(?<=\()\d+|\d+(?=\))', line)]
                width = x_max - x_min
                height = y_max - y_min
                values = map(int, (x_min, y_min, width, height))
                counter += 1
                coordinates.append(values)
    return coordinates



def is_interception(first, second):
    if first[0] > second[0]:
        if second[0] + second[2] < first[0]:
            return False
    else:
        if first[0] + first[2] < second[0]:
            return False
    if first[1] > second[1]:
        if second[1] + second[3] < first[1]:
            return False
    else:
        if first[1] + first[3] < second[1]:
            return False
    return True
