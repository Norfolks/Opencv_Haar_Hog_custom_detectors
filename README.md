# Opencv Haar Hog custom detectors

Nerual networks become most accurate standart for object detection and classification nowday.
However, they still require much cpu power, RAM and time to train and real-time video processing. So, for simple tasks you could want to use old techincs as Haar cascade or svm trained on HOG features. This work presents the way you can do it.

You can go throught this project and make similar things for your dataset.

# Get data first

Download dataset:
```
wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar
```
After downloading create links of Test and Train data to the project folder.


# How to train your classifier

To train Haar look at usefull links part

To train HOG SVM run train_hog.py

# Check classifiers

Run Haar or Hog detector(haar_detector.py/hog_detector.py)

This files will show stats

# Files description

cascade*.xml - files are trained haar cascades

*.vec - created samples for haar cascade training wiht opencv_createsamples

# Usefull links:
[1] http://techcave.ru/posts/55-obuchenie-kaskadnogo-klassifikatora-v-opencv-opencv-traincascade-opencv-createsamples.html - how to train haar cascade


