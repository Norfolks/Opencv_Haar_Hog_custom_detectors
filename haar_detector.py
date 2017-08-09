import cv2
import re
import numpy as np
from utils import *

cascade = cv2.CascadeClassifier('cascade.xml')

# Positive images
with open('Test/pos.lst') as test_images, open('Test/annotations.lst') as annotations:
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    for test_image_path, ann_path in zip(test_images, annotations):
        test_image_path = test_image_path[:-1]
        ann_path = ann_path[:-1]
        img = cv2.imread(test_image_path)
        detections = cascade.detectMultiScale(img)
        detections = non_max_suppression_fast(detections, 0.5)
        true_coordinates = get_list_of_annotation_data(ann_path)
        good_one = False
        for (x_t, y_t, w_t, h_t) in true_coordinates:
            best_match = -1
            best_match_index = -1
            for index, (x_d, y_d, w_d, h_d) in enumerate(detections):
                if not is_interception((x_t, y_t, w_t, h_t), (x_d, y_d, w_d, h_d)):
                    continue
                xs = sorted([x_d, x_d + w_d, x_t, x_t + w_t])
                ys = sorted([y_d, y_d + h_d, y_t, y_t + h_t])
                interception_square = (xs[2] - xs[1]) * (ys[2] - ys[1])
                group_square = int(w_d) * int(h_d) + int(w_t) * int(h_t) - interception_square
                bd_eval = interception_square / group_square
                if bd_eval > best_match and bd_eval > 0.5:
                    best_match_index = index
                    best_match = bd_eval
            if best_match_index > -1:
                np.delete(detections, best_match_index)
                # del (detections[best_match_index])
                true_positives += 1
                good_one = True
            else:
                false_negatives += 1

        false_positives += len(detections)
        if good_one:
            result = non_max_suppression_fast(cascade.detectMultiScale(img), 0.5)
            for (x, y, w, h) in result:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('image', img)
            key_pressed = cv2.waitKey(0)

# Negative Images
with open('Test/neg.lst') as test_images:
    for test_image_path in test_images:
        test_image_path = test_image_path[:-1]
        img = cv2.imread(test_image_path)
        detections = cascade.detectMultiScale(img)
        false_positives += len(detections)

print('True positives: ' + str(true_positives))
print('False negatives: ' + str(false_negatives))
print('False positives: ' + str(false_positives))
print('Precision: ' + str(true_positives / (true_positives + false_positives)))
print('Recall: ' + str(true_positives / (true_positives + false_negatives)))
