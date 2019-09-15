# Bottle Checker
## Overview
Bottle Checker is an app that detects and sorts PET bottle images.

## Description
I created the Bottle Checker as a demonstration of object detection and image recognition.
  

You can upload a PET bottle image. Also, you can have several PET bottles in one image. Bottle Checker detects the position of the label on the PET bottle and classifies it by machine learning. Then, the kind of PET bottle and the determination probability are output.
  

I created two machine learning models, an object detection model and an image recognition model. 
The object detection model was implemented with OpenCV, and the image recognition model was implemented with Chainer. 
And they trained more than 4,000 images each.

## Demo
This is the top page. You can upload a photo of plastic bottles.

![bottleCheckerSample2](https://github.com/takitaki7474/algorithm-research/blob/images/bottleChecker_images/bottle_checker_sample2.png)

![bottleCheckerDemo](https://github.com/takitaki7474/algorithm-research/blob/master/gifs_and_images/bottle_checker.gif)
  

To detect plastic bottles from an uploaded photo and to guess which class the plastic bottle belongs to.  
Further, images after detection and the probability are displayed. 

![bottleCheckerSample1](https://github.com/takitaki7474/algorithm-research/blob/images/bottleChecker_images/bottle_checker_sample1.png)
