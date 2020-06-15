# License-Plate-Recognition

### Demo:
##### For videos
To run demo on sample videos with default arguments- "python demo_video.py"
else run- "python demo_video.py --input inputs/demo1.mp4 --output results/output1.avi --size 608"

### How to train:
1. Download dataset zip from this link - https://www.kaggle.com/tustunkok/license-plate-detection/data
2. Create a folder named 'dataset' inside 'data' and extract the content of zip into 'dataset' folder.
3. Go to 'data' folder and run the given command on cmd - "python augmentation.py". After augmentation number of images will be 2154.
4. Now to train darknet YOLOv4 on the dataset, follow the steps given here - https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
