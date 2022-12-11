[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"
[image4]: ./images/index.png "Web app's index page"
[image5]: ./images/predict.png "Predict result plot"

## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, your algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

Along with exploring state-of-the-art CNN models for classification, you will make important design decisions about the user experience for your app.  Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  Your imperfect solution will nonetheless create a fun user experience!

## Project Instructions

### Start from scratch

Please follow original repo here: https://github.com/udacity/dog-project.git

### To run demo web-app

- Install the dependent packages in the requirement_lastest.txt file.
- Start flask app located in folder dog-breeds-detector/web-app/manage.py with command

'''
python manage.py
'''

- access to upload files at 127.0.0.1:5000 (can upload multiple files at once)
![Web app's index page][image4]
- tap the dog breed prediction button to process. The results are similar to the following:
The left is the original image, the right is the prediction result.
- If it is identified as a person, it will analyze which dog breed this person has similarity to.
- If recognized as a dog, return the breed and image of the training set
- If detected neither human nor dog, return the original image on the right.
![Predict result plot][image5]

### Credit: 
File upload function refer to the library and examples of Flask-Dropzone
