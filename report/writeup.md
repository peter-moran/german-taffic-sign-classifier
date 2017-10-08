#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./example_counts.png "Sign Example Counts"
[image2]: ./before-pp.png "Example image before preprocessing"
[image3]: ./after-pp.png "Example image after preprocessing"
[image4]: ./error-convergence.png "Validation error convergence over each epoch"
[image5]: ../test_signs/id0.jpg "Sign of ID 0"
[image6]: ../test_signs/id9.jpg "Sign of ID 9"
[image7]: ../test_signs/id14.jpg "Sign of ID 14"
[image8]: ../test_signs/id25.jpg "Sign of ID 25"
[image9]: ../test_signs/id31.jpg "Sign of ID 31"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it! and here is a link to my [project code](../Traffic_Sign_Classifier.html).

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set.

I used the numpy to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32 x 32 x 3).
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset.

This chart shows the number of images in both the training and validation set. This exposes the fact that there are many more images for some sign classes. It also shows that the validation and training set have similar ratios for number of each image type.

![alt text][image1]

In addition, the project code shows an example image of each sign, labeled with both ID and text name. For brevity, it is not included here, but these visualizations were helpful as a reference and for sanity checking labels.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data.

To pre-process the images, all I did was min-max normalize them in order to help the neural network generalize. I tested additional changes, such as converting to HSV color space and cropping -- which I could turn on and off by binary flags, but ultimately I found these did not improve performance. These steps are turned off for now.

Here is an example of a traffic sign image before and after preprocessing.

![alt text][image2]
![alt text][image3]

I attempted creating additional data, but ultimately had too much trouble doing it for the time I had. The code for this was non-functional and removed.

####2. Describe what your final model architecture looks like.

My model is based on a series of convolutions that vary in output size only, followed by fully connected layers that also vary in output size only. Each convolution layer ends with max pooling, except for the first layer. All layers except the last layer has RELU activation and, during training, 50% dropout.

The result is a model with 1,099,071 parameters.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32 	|
| RELU					|												|
| Dropout				| 50% dropout applied during training			|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x48					|
| Dropout				| 50% dropout applied during training			|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 12x12x144 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x48					|
| Dropout				| 50% dropout applied during training			|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 4x4x432	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x432					|
| Dropout				| 50% dropout applied during training			|
| Fully connected		| output 512   									|
| RELU					|												|
| Dropout				| 50% dropout applied during training			|
| Fully connected		| output 256   									|
| RELU					|												|
| Dropout				| 50% dropout applied during training			|
| Fully connected		| output 43   									|

####3. Describe how you trained your model.

To train the model, I used the Adam Optimizer on the cross-entropy error between the softmax of the logit output of the neural network with one-hot labels.

Weights were initialized with normal random variables with a standard deviation of 0.01. The network was trained over 45 epochs, using a batch size of 512 and an initial learning rate of 0.001.

These hyper-parameters were chosen by tuning them over multiple runs, starting with standard "safe guesses".

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:

* Validation set accuracy of 97.8% 
* Test set accuracy of 97.4%

Below is the validation error convergence over each epoch.

![alt text][image4]

**What was the first architecture that was tried and why was it chosen?**

I started off with a basic LeNet structure as shown in the lectures. Then, because I was using color images and had more classes, I decided to widen each layer, making it so each layer would use more filters and produce an output with more feature maps. This increases the number of parameters of my model, and added to its ability to capture greater complexity, thus increasing the accuracy to over 90%.

Next I added dropout. This brought me over the required 93% accuracy, but I desired to increase it further.

By using helper functions that created convolutional and fully connected layers with the correct input sizes and options, I was then able to test many different layer combinations, using layers with and without pooling, changing the kernel sizes, changing output sizes, dropout percentage, and adding new layers automatically with ease similar to Keras. I also kept an eye on the number of parameters and used charts of performance over epochs to help inform my changes.

I found that adding another convolution layer and decreasing my kernel helped me get above 96% validation accuracy, leading me to my current design.

For the most part, I did not encounter overfitting, where the validation accuracy decreases after too many epochs. For the most part, the network would quickly learn and then saturate at some final level of accuracy, after which I would terminate it.

**What were some problems with the initial architecture?**

Along the way, I tried a few different ideas. I attempted to pre-calculate edges for the network, using canny edge detection and appending it to the RGB image (turning it into an input with a depth of 4). This did not change much and was scrapped. I also tried generating new data, but the result was buggy, I ran out of time, and it was scrapped. I tested changing color space to HSV, but this did not do much. I also tried cropping, but this either decreased performance or did not effect it.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report.

Here are five German traffic signs that I found on the web, resized to 32x32 as given to the network:

![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]
![alt text][image9]

All these images were well lit, but some, such as ID 0 and ID 14, did have busy backgrounds. I chose these signs specifically because I felt they were some of the more difficult signs in the dataset and was interested in what the secondary guesses by the network were.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

Here are the results of the top prediction of the network for each image, along with the subsequent 4 best guesses.

```
Input 0 (ID 0, Speed limit (20km/h)) was believed to be
	 95.758 % ->  0, Speed limit (20km/h)
	  2.267 % ->  8, Speed limit (120km/h)
	  1.708 % ->  1, Speed limit (30km/h)
	  0.264 % ->  4, Speed limit (70km/h)
	  0.001 % ->  7, Speed limit (100km/h)
Input 1 (ID 9, No passing) was believed to be
	100.000 % ->  9, No passing
	3.5e-07 % -> 10, No passing for vehicles over 3.5 metric tons
	7.1e-08 % -> 16, Vehicles over 3.5 metric tons prohibited
	1.2e-08 % -> 17, No entry
	5.2e-09 % -> 41, End of no passing
Input 2 (ID 14, Stop) was believed to be
	 99.990 % -> 14, Stop
	  0.008 % -> 17, No entry
	  0.002 % ->  1, Speed limit (30km/h)
	3.2e-04 % -> 25, Road work
	1.6e-04 % -> 26, Traffic signals
Input 3 (ID 25, Road work) was believed to be
	 99.999 % -> 25, Road work
	2.7e-04 % -> 22, Bumpy road
	2.4e-04 % -> 29, Bicycles crossing
	2.1e-05 % -> 31, Wild animals crossing
	3.4e-06 % -> 38, Keep right
Input 4 (ID 31, Wild animals crossing) was believed to be
	 98.159 % -> 31, Wild animals crossing
	  1.225 % -> 23, Slippery road
	  0.564 % -> 21, Double curve
	  0.030 % -> 19, Dangerous curve to the left
	  0.008 % -> 20, Dangerous curve to the right
```

As you can see, the network had a 100% accuracy on this test set. What I found even more interesting was the very high confidence of the model on many of the images. This is in line with the test set's accuracy, as 97% is near 100% and we have few samples.

The most difficult images were the 20 km/h Speed limit sign and the Wild animals crossing sign. This makes sense, as other signs in the dataset look very similar, with the only difference being slight details in the center of the signs.


