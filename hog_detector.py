import cv2
import numpy as np
from sklearn.externals import joblib
import pickle
from utils import *


svm = pickle.load(open("svm.pickle", 'rb'))
model = joblib.load('svm.bin')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(
    # np.array(svm)
    # cv2.HOGDescriptor_getDefaultPeopleDetector()
    model.coef_.reshape(3780, 1)

)

# Positive images
with open('Test/pos.lst') as test_images, open('Test/annotations.lst') as annotations:
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    for test_image_path, ann_path in zip(test_images, annotations):
        test_image_path = test_image_path[:-1]
        ann_path = ann_path[:-1]
        img = cv2.imread(test_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = hog.detectMultiScale(img, finalThreshold=2)
        detections = detections[0]
        detections = non_max_suppression_fast(detections, 0.1)
        true_coordinates = get_list_of_annotation_data(ann_path)
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
                true_positives += 1
            else:
                false_negatives += 1

        false_positives += len(detections)

# Negative Images
with open('Test/neg.lst') as test_images:
    for test_image_path in test_images:
        test_image_path = test_image_path[:-1]
        img = cv2.imread(test_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = hog.detectMultiScale(img, finalThreshold=2)[0]
        detections = non_max_suppression_fast(detections, 0.1)
        false_positives += len(detections)

print('True positives: ' + str(true_positives))
print('False negatives: ' + str(false_negatives))
print('False positives: ' + str(false_positives))
print('Precision: ' + str(true_positives / (true_positives + false_positives)))
print('Recall: ' + str(true_positives / (true_positives + false_negatives)))
