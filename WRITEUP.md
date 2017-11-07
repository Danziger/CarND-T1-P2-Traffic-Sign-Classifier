CarND · T1 · P2 · Traffic Sign Classifier
=========================================


[//]: # (Image References)

[image1]: ./output/images/001%20-%20All%20Signs.png "All Signs"
[image2]: ./output/images/002%20-%20Initial%20Distribution.png "Initial Distribution"
[image3]: ./output/images/003%20-%20Preprocessing%20Combinations.png "Preprocessing Combinations"
[image4]: ./output/images/004%20-%20Augmentation%20Examples.png "Augmentation Examples"
[image5]: ./output/images/005%20-%20Augmented%20Distribution.png "Augmented Distribution"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


Project Goals
-------------

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


Rubric Points
-------------

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


### WRITEUP / README

#### 1. Provide a WRITEUP / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

The project code can be found [here](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) and the README of this repo provides a description of the project's structure.


### DATA SET SUMMARY & EXPLORATION

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set.

First, the absolute and relative sizes of each data set:

|   DATA SET |  UNITS |        % |
|------------|--------|----------|
|   TRAINING |  34799 |  67.13 % |
| VALIDATION |   4410 |   8.51 % |
|    TESTING |  12630 |  24.36 % |
|      TOTAL |  51839 | 100.00 % |

Then, the size of each image: `32 px WIDTH × 32 px HEIGHT × 3 CHANNELS`.

Lastly, the number of unique classes/labels in the data set, as well as a list of all of them:

| ID | SIGN NAME | 
|----|-----------|
|  0 | Speed limit (20km/h) |
|  1 | Speed limit (30km/h) |
|  2 | Speed limit (50km/h) |
|  3 | Speed limit (60km/h) |
|  4 | Speed limit (70km/h) |
|  5 | Speed limit (80km/h) |
|  6 | End of speed limit (80km/h) |
|  7 | Speed limit (100km/h) |
|  8 | Speed limit (120km/h) |
|  9 | No passing |
| 10 | No passing for vehicles over 3.5 metric tons |
| 11 | Right-of-way at the next intersection |
| 12 | Priority road |
| 13 | Yield |
| 14 | Stop |
| 15 | No vehicles |
| 16 | Vehicles over 3.5 metric tons prohibited |
| 17 | No entry |
| 18 | General caution |
| 19 | Dangerous curve to the left |
| 20 | Dangerous curve to the right |
| 21 | Double curve |
| 22 | Bumpy road |
| 23 | Slippery road |
| 24 | Road narrows on the right |
| 25 | Road work |
| 26 | Traffic signals |
| 27 | Pedestrians |
| 28 | Children crossing |
| 29 | Bicycles crossing |
| 30 | Beware of ice/snow |
| 31 | Wild animals crossing |
| 32 | End of all speed and passing limits |
| 33 | Turn right ahead |
| 34 | Turn left ahead |
| 35 | Ahead only |
| 36 | Go straight or right |
| 37 | Go straight or left |
| 38 | Keep right |
| 39 | Keep left |
| 40 | Roundabout mandatory |
| 41 | End of no passing |
| 42 | End of no passing by vehicles over 3.5 metric tons |


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

First, I plotted one image per class:

![All Signs][image1]

Then, I created a bar chart showing the classes distribution in each data set:

![Initial Distribution][image2]


### DESIGN AND TEST A MODEL ARCHITECTURE

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

#### DATA PREPROCESSING IMPLEMENTATION

The preprocessing techniques I considered are:

- Grayscaling, as grayscale images are faster to process, thus speeding up learning, than color ones (one channel VS three).

    Also, even though the color information might be very useful to identify some signs, if used, it's important to augment the data set with "incorrectly colored" or "artificially tinted" images, as some real-world illumination conditions, such us really bright sunlight, headlights reflections, tail lights colorful illumination, neon/led signs, traffic lights... will generate weirdly colored inputs.
  
- Augmenting contrast (on the Y channel of color images) or histogram equalization (on grayscal images), to clearly distinguish shapes and figures, specially on really dark or bright images.

- Sharpening (using Gaussian Blur), again, to clearly distinguish shapes, figures and borders.

- Cropping, to remove irrelevant information, as most signs don't extend to the border of the image, but keep a generous margin around instead.

Note normalization is not mentioned here because, as I will also augment the training set, that step will be applied after that on the whole augmented training set, instead of just on the original images.

In order to decide which of this options to apply and how to combine them, I plotted different possible combinations and took the one where the signs were sharper and easier to distinguish, which in my opinion is the last column, that is, `GRAYSCALE ∘ CONTRAST ∘ SHARPEN ∘ CROP`:

![Preprocessing Combinations][image3]

Note this preprocessing is applied to all 3 datasets (training, validation and test).


#### DATA PREPROCESSING POSSIBLE IMPROVEMENTS

Instead of a small cropping of the image, a gradient region filter (vignette) might be used to filter out irrelevant data progressively (the closer to the border of the image, the less relevant it's likely to be).

Some other image processing algorithms might be used. In fact, I'm really curious to know if a canny edge detection processed image (just edges, without the original image) would yield better results.


##### DATA SET AUGMENTATION IMPLEMENTATION

Next, we can see from the initial classes distributionthat some classes are clearly underrepresented, such as class 0 (Speed limit (20km/h)), so I decided to augment the training data set in order to give the network enough chances to learn the features of the most scarce classes.

To add more data to the the data set, I did the following steps:

-  For a given class, calculate the desired target `T` of ocurrences, which is 1500 if there are less than 1500 ocurrences originally `O`, or 2100 otherwise. Therefore, I'm agumenting all the classes, even those that already have a decent number of examples, as the more and more variate data we have, the better our algorithm will be.

    Note I still want to keep the difference in occurrences between the most common  and the least common classes, as I think it makes sense to take into consideration the natural ocurrence of signs in the real world.
  
- Next, I took a randonm set of `R = T - O` images from that given class that I will use as base images to generate new ones. Note the same image can be used more than once (randomly), but specially when `O < R`.

- To those selected images, I apply a randomized set of transformations, which might include rotation (of -5, 5, -10, 10, -15, 15, -20, 20, -25 or 25 deg), sharpening and/or clipping.

Below you can see some examples of these randomly generated images are:

![Augmentation Examples][image4]

After augmenting all the classes, the new classes distribution looks like this:

![Augmented Distribution][image5]


##### DATA SET AUGMENTATION POSSIBLE IMPROVEMENTS

A more methodical approach might be used to generate (evem more and more variated) additional examples instead of relaying so much on randomness.

Also, a matrix transform could be used instead of a simple 2D rotation to achieve 3D rotation/projection.

Saturation/lightness transformations of all kind might be used as well: uniform transformations, randomly-shaped filters (one half of the image gets a filter, the other half another one, or the opposite...), gradient filters resembling spot lights or reflexions...

Additionally, random noise/marks could be introduced or drawn on top of the images resembling tree branches, paint deterioration, corrosion, rain, fog...


#### DATA SET NORMALIZATION

Lastly, all 3 datasets are normalized to speed up convergence and ensure all feature are equally represented in the resulting model.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


