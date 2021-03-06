# Bottle Checker
## Overview
Bottle Checker is an app that detects and classifies PET bottle images by machine learning.

## Description
I created the Bottle Checker as a demonstration of object detection and image recognition.
  

You can upload a PET bottle image. Also, you can have several PET bottles in one image. Bottle Checker detects the position of the label on the PET bottle and classifies it by machine learning. Then, detected PET bottle labels are outlined in red. And  the kind of PET bottle and the determination probability are output. 
  

I created two machine learning models, an object detection model and an image recognition model. 
The object detection model was implemented with OpenCV, and the image recognition model was implemented with Chainer. 
And they trained more than 4,000 images each.

## Demo

![bottleCheckerDemo](https://github.com/takitaki7474/bottle_checker/blob/demo-images/demo-images/bottle_checker.gif)
  

![bottleCheckerSample1](https://github.com/takitaki7474/bottle_checker/blob/demo-images/demo-images/bottle_checker_sample1.png)
