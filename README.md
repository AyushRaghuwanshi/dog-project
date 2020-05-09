# Dog Breed Classifier

[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview :
In this project, we  learned how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, your algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

## My Goals & Motivation :
- to create image clssification model from scrach and from pretrained models.
- to compare the results between them.
- to explore different techniques for data agumentation.
- to get an understanding opencv and haarcascade features and pytorch framework.

## Files Description :
### dog_app.ipynb :
	- It contains all the code which were used to create and train the model from scrach and from pretraind model. It also contains the testing of those trained models.
### haarcascades :
	- This folder contains haarcasecade features of front face which are used to detect front face in an image.
### images :
	- This folder contains some testing images which can be used to test the model.
### saved_models :
	- trained model are going to save here.


## Libraries used :
- pytorch = As a deep learning framework.
- numpy = for array manipulation.
- opencv(cv2) = to load images and their manipulation.
- matplotlib = for plot an image.
- torchvision = To get pretarined model and image transformation and loading.

## Results of the analysis

As we can see the difference in the accuracy of pre-trained model and our model is huge we can see how pre-trained model can make our life easier with few lines of codes however it comes with a cost of heavy computation and complexity.
As we can see the choice of pre-trained model proved to be a good choice because we were able to get the accuracy over 80%.

### other details can be found here(Blog Post Link)
https://medium.com/@ayushraghuwanshi98/dog-breed-classifier-dsnd-capstone-project-9eb710431be0

## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/udacity/dog-project.git
cd dog-project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

