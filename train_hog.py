import cv2
import re
import numpy as np

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.externals import joblib

hog = cv2.HOGDescriptor()
reshape_value = 3780
init_widht = 64
init_height = 128
locations = ((init_widht, init_height),)


def get_list_of_annotation_data(ann_path):
    with open(ann_path, 'r', encoding='ISO-8859-15') as annotation_description:
        counter = 0
        coordinates = list()
        for line in annotation_description:
            if line.startswith('Bounding'):
                x_min, y_min, x_max, y_max = [int(el) for el in re.findall(r'(?<=\()\d+|\d+(?=\))', line)]
                width = x_max - x_min
                height = y_max - y_min
                values = list(map(int, (x_min, y_min, width, height)))
                counter += 1
                coordinates.append(values)
    return coordinates


# Init train data
train_data = []
label_data = []
with open('Train/pos.lst') as positives_file, \
        open('Train/annotations.lst', 'r') as annotations_list:
    count_positive = 0
    for train_image_path, ann_path in zip(positives_file, annotations_list):
        train_image_path = train_image_path[:-1]
        ann_path = ann_path[:-1]
        annotation_for_image = get_list_of_annotation_data(ann_path)
        image = cv2.imread(train_image_path)
        for val in annotation_for_image:
            cropped_image = image[val[1]:val[1] + val[3], val[0]:val[0] + val[2]]
            height_ratio = cropped_image.shape[0] / init_height
            width_ratio = cropped_image.shape[1] / init_widht
            if cropped_image.shape[1] / height_ratio < init_widht:
                cropped_image = cv2.resize(cropped_image, (
                int(cropped_image.shape[1] / width_ratio), int(cropped_image.shape[0] / width_ratio)))
            else:
                cropped_image = cv2.resize(cropped_image, (
                int(cropped_image.shape[1] / height_ratio), int(cropped_image.shape[0] / height_ratio)))
            cropped_image = cropped_image[0:init_height, 0:init_widht]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            hog_features = hog.compute(cropped_image)
            if hog_features[0] == 0:
                continue
            train_data.append(hog_features.reshape(reshape_value))
            label_data.append(1)
            count_positive += 1

with open('Train/neg.lst') as negative_file:
    count_false = 0
    for train_image_path in negative_file:
        train_image_path = train_image_path[:-1]
        image = cv2.imread(train_image_path)
        image = image[(image.shape[0] - init_height) // 2:(image.shape[0] + init_height) // 2,
                (image.shape[1] - init_widht) // 2:(image.shape[1] + init_widht) // 2]
        hog_features = hog.compute(image)
        if hog_features[0] == 0:
            continue
        train_data.append(hog_features.reshape(reshape_value))
        label_data.append(0)
        count_false += 1

print('Positive samples: ' + str(count_positive))
print('Negative samples: ' + str(count_false))
train_data = np.array(train_data)
label_data = np.array(label_data)
print('Start data shuffling...')
shuffle(train_data, label_data)

# Sklearn SVM
model = SVC(C=1, kernel='linear')
print('Start SVM training')
model.fit(train_data, label_data)
joblib.dump(model, 'svm.bin')

# OpenCV SVM
import xml.etree.ElementTree as ET
import pickle

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1000, 1.e-03))
svm.train(train_data, cv2.ml.ROW_SAMPLE, label_data)

svm.save("svm.xml")
tree = ET.parse('svm.xml')
root = tree.getroot()
# now this is really dirty, but after ~3h of fighting OpenCV its what happens :-)
SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0]
rho = float(root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text)
svmvec = [float(x) for x in re.sub('\s+', ' ', SVs.text).strip().split(' ')]
svmvec.append(-rho)
pickle.dump(svmvec, open("svm.pickle", 'wb'))
