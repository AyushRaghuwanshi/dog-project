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
	- It contains all the code which were used to create and train the model from scrach and from pretraind model. It also contains the testing of those trained models.<br>
	most of the code was provided by udacity and that notebook were broken down into separate steps which were :<br>
Step 0: Import Datasets<br>
Step 1: Detect Humans<br>
Step 2: Detect Dogs<br>
Step 3: Create a CNN to Classify Dog Breeds (from Scratch)<br>
Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)<br>
Step 5: Write your Algorithm<br>
Step 6: Test Your Algorithm<br>
Step 0: Import Datasets<br><br>
In this code cell, we save the file paths for both the human (LFW) dataset and dog dataset in the NumPy arrays human_files and dog_files.<br><br>
Step 1: Detect Humans<br>
As we know OpenCV provides many pre-trained face detectors, stored as XML files. I had downloaded one of these detectors and stored it in the haarcascades directory. I used that haar cascade feature to detect faces in an image.<br>
Before using any of the face detectors, it is standard procedure to convert the images to grayscale. The detectMultiScale function executes the classifier stored in face_cascade and takes the grayscale image as a parameter.<br>
faces is a numpy array of detected faces, where each row corresponds to a detected face. Each detected face is a 1D array with four entries that specifies the bounding box of the detected face. The first two entries in the array (extracted in the above code as x and y) specify the horizontal and vertical positions of the top left corner of the bounding box. The last two entries in the array (extracted here as w and h) specify the width and height of the box.<br><br>
Step 2: Detect Dogs<br>
In this section, I used a pre-trained VGG-16 model to detect dogs in images.<br>
the VGG-16 model, along with weights that have been trained on ImageNet, a very large, very popular dataset used for image classification and other vision tasks. ImageNet contains over 10 million URLs, each linking to an image containing an object from one of 1000 categories.<br>
I noticed that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151–268, inclusive, to include all categories from 'Chihuahua' to 'Mexican hairless'. Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained VGG-16 model, we need only check if the pre-trained model predicts an index between 151 and 268 (inclusive).<br><br>
Step 3: Create a CNN to Classify Dog Breeds (from Scratch)<br>
In this part, I wrote three separate data loaders for the training, validation, and test datasets of dog images (located at dog_images/train, dog_images/valid, and dog_images/test, respectively). I used data augmentation like random horizontal flip, random rotation.after that I converted the image into a tensor and normalize by standard mean and sd.
then I created a CNN to classify dog breed<br>
In this architecture batch normalization layer could have been used to fast learning and dropout prob could have been set to 0.5 to avoid overfitting.<br>
I define CrossEntropyLoss as loss function and sgd optimiser. Adam optimizer could have been the better choice.<br>
then I trained model for 70 epochs.<br>
then I tested that and got 10% accuracy.<br><br>
Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)<br>
As we saw I got a 10% accuracy from our model so I went for the transfer learning.<br>
As we know our data set is similar to some classes of imagenet dataset however the size of dataset is not that large so we can use the initial layers of pre-trained with their trained weights so the initial layer acts as a feature detector and then on the top of that we can trained top fully connected layers.<br>
We know that the imagenet has 1000 classes so the pre-trained model trained on top of that have 100 outputs on last layer . to map the output to our labels size I changed the output layer of the model.<br>
I choose VGG16 as pre-trained model and did the following steps to train that =
First I load the model with its pre-trained weights.<br>
Then I set its features layers require grad to false because we did not want to train them.<br>
Then I changed its final layer to change its output dimension according to out dataset.<br>
I choose CrossEntropyLoss as loss function and sgd for optimizer and trained the model and got 81% accuracy this time.<br><br>
Step 5: Write your Algorithm<br>
In this part I had to write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither. Then,<br>
if a dog is detected in the image, return the predicted breed.<br>
if a human is detected in the image, return the resembling dog breed.<br>
if neither is detected in the image, provide an output that indicates an error.<br>
In this, I used vgg16 pre-trained for dog detection and if found then used our version vgg16 trained model to identify the breed<br>
used face detector to detect human face if found then used our version of vgg16 to identity the resembling breed.<br><br>
Step 6: Test Your Algorithm<br>
The output is much better than I expected. it gives the same breed type to a human who looks the same. even it gives the correct breed of dog in a picture where human was with her.<br>
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

