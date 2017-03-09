## Vehicle Detection
My Vehicle Detection and Tracking project for Udacity Self Driving Car Nanodegree. This is Project 5. 

[//]: # (Image References)
[sample_car_notcar.png]: ./output_images/sample_car_notcar.png
[HOG_features_YCrCb.png]: ./output_images/HOG_features_YCrCb.png
[before_after_normalization.png]: ./output_images/before_after_normalization.png
[sliding_windows_6.png.png]: ./output_images/sliding_windows_6.png
[w_heatmap_testimage6.png]: ./output_images/w_heatmap_testimage6.png
[w_heatmap_testimage6_labels.png]: ./output_images/w_heatmap_testimage6_labels.png
[video_sample_image.png]: ./output_images/video_sample_image.png

[//]: # ![alt text][A.png]

Overview
---
In this project, the goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4). 

Data source
---
Data is provided with a labeled dataset and the job is to decide what features to extract, then train a classifier and ultimately track vehicles in a video stream. Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

Basic steps of this project:
Step1: Data exploration
There are 8792 number of samples in cars set, 8968 number of samples in notcars set. 
We plot example car and not car image to have a general idea about the dataset. 

![alt text][sample_car_notcar.png]

Step2: Extract image features to identify car and not car content.  
I used HoG (Histogram of oriented gradients), spatial_features, color histogram to extract image features. 
- spatial feature (reduce the image size): 
`spatial_features = bin_spatial(feature_image, size=spatial_size)`
- color histogram feature (np.histogram on color image):
`hist_features = color_hist(feature_image, nbins=hist_bins)`
- HoG feature (get histogram of oriented gradient after converting image to certain color space onto every block and cells, hog is in `skimage.feature` package):
`features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=vis, feature_vector=feature_vec)`

![alt text][HOG_features_YCrCb.png]

Step3: Normalize features and use machine learning method (SVM in this task) to train dataset using optimized parameters for extracting features
I used StandScaler() to normalize extracted features. And we could see before and after normalization:

![alt text][before_after_normalization.png]

In this project linear SVM is used and provided an acceptable testing accuracy. 
80% of dataset has been used a training set and 20% as testing set. We shuffle data before training to avoid algortihm remembering the ordering. I tried different combination of HoG settings as well color space for feature extraction. The different option testing accuracies are included below: 

| Option | Color Space | Spatial_size |Hist_bins|  Orientations | Pixels_per_cell | Cells_per_block | HOG channel | Time for Train | Time to Predict first 100 | Testing Accuracy |
|:------:|:-----------:|:------------:|:-------:|:-------------:|:---------------:|----------------:|------------:|---------------:|--------------------------:|-----------------:|
| 0   |  RGB   | (16,16) | 32 | 8 | 7 | 2 | All | 28.16s | 0.003s | 97.4% |
| 1   |  HSV   | (16,16) | 32 | 8 | 7 | 2 | All | 7.64s  | 0.003s | 98.7% |
| 2   |  HLS   | (16,16) | 32 | 9 | 8 | 2 | All | 10.5s  | 0.003s | 94.4% |
| 3   |  YCrCb | (32,32) | 32 | 9 | 7 | 2 | All | 8.29s  | 0.003s | 98.9% |
| 4   |  YCrCb | (32,32) | 16 | 8 | 7 | 2 | All | 8.39s  | 0.003s | 98.9% |
| 5   |  YCrCb | (32,32) | 32 | 8 | 7 | 2 | All | 21.4s  | 0.003s | 99.1% |
| 6   |  YCrCb | (16,16) | 32 | 8 | 8 | 2 | All | 15.3s  | 0.003s | 98.7% |
| 7   |  YCrCb | (16,16) | 32 | 8 | 7 | 2 | All | 18.6s  | 0.003s | 98.8% |

Hence we select option 5 which use ColorSpace of YCrCb and Spatial_size (32,32), 32 Hist bins, orient 8, Pixels_per_cell 7, Cell_per_block of 2, HOG channel All. 

Step4: Build a sliding window technique to identify car/not car content from a large image 
We used 6 different sizes windows to detect car or not car content. X direction cover from leftmost to rightmost, while y direction covers from near front of driving car to the horizon. We use overlay of 75% as a default. We have smaller windows mainly at the horizon for smaller car images as they will appear far away. Example detected window of car and all the sliding windows are below: 

![alt text][sliding_windows_6.png]

Step5: Use method like heatmap of recurring detections to avoid false positive/negative classifications
As we could see there are some false positive windows which identify car image on something which are not cars. I use heatmap for the recurring detection and apply threshold to filter out low recurrence which are more likely false positive. 
Example of test image under heatmap is like 

![alt text][w_heatmap_testimage6.png]

![alt text][w_heatmap_testimage6_labels.png]

Step6: Build video pipeline to process streaming videos with bounding box to detect and track vehicles. I apply Advanced Lane Finding with vehicle detection and tracking together. 

![alt text][video_sample_image.png]


### The repository contains: 
* Vehicle_Detection_And_Tracking_MZ.ipynb (main script)
* CameraData.p (Camera calibration data from previous project using chessboard images)
* data_carsnotcars.p (Car and not a car image data, raw images are not included due to large size, but link has been provided above in Data source section)
* ClassifierData.p, ProcessedData.p (temperaly output data files for easy resume work)
* lesson_functions.py (functions build in course which helps extract image car features)
* folder `output_images` (output images)
* 

### Discussion
As we could see from the output video, the vehicle detection is still not perfect yet as it sometimes gives pretty large bounding box onto the car which in practice means low accuracy of knowing where exactly the car is. An improvement could be using smaller sliding windows or apply higher threshold on heatmap. 
Another further possible improvement is to achieve better testing accuracy using deep learning method instead of linear SVM. However using deep learning may require more carefulness in terms of responsiveness in real time driving.  
