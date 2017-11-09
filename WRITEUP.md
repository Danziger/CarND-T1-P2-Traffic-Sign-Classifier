CarND · T1 · P2 · Traffic Sign Classifier Writeup
=================================================


[//]: # (Image References)

[image1]: ./output/images/001%20-%20All%20Signs.png "All Signs"
[image2]: ./output/images/002%20-%20Initial%20Distribution.png "Initial Distribution"
[image3]: ./output/images/003%20-%20Preprocessing%20Combinations.png "Preprocessing Combinations"
[image4]: ./output/images/004%20-%20Augmentation%20Examples.png "Augmentation Examples"
[image5]: ./output/images/005%20-%20Augmented%20Distribution.png "Augmented Distribution"
[image6]: ./output/images/006%20-%20Accuracy%20plot.png "Accuracy Plot"
[image7]: ./output/images/007%20-%20Probability%20Plot.png "Probability Plot"
[image8]: ./output/images/008%20-%20Feature%20Map%20Yield.png "Feature Map Yield"
[image9]: ./output/images/009%20-%20Feature%20Map%20Stop.png "Feature Map Stop"
[image10]: ./output/images/010%20-%20Feature%20Map%20Better%20Stop.png "Feature Map Better Stop"

[sign1]: ./input/images/resized/001%20-%20Yield.jpg "Yield"
[sign2]: ./input/images/resized/002%20-%20Stop.jpg "Stop"
[sign3]: ./input/images/resized/003%20-%20Road%20Work.jpg "Road Work"
[sign4]: ./input/images/resized/004%20-%20Priority%20Road.jpg "Priority Road"
[sign5]: ./input/images/resized/005%20-%20Speed%20limit%20(30km:h).jpg "Speed limit (30km/h)"
[sign6]: ./input/images/resized/006%20-%20Better%20Stop.jpg "Better Stop"

 
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

Instead of a small cropping of the image, a gradient region filter (vignette) might be used to filter out irrelevant data progressively (the closer to the border of the image, the less relevant it's likely to be). Also, Hough transforms might be used to detect arbitrary shapes (circle, triangle, squares or hexagons) and filter out the background of the image before applying the other preprocessing operations.

Another possible image processing algorithm that we might want to apply is canny edge detection.


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


##### DATA SET NORMALIZATION

Lastly, all 3 datasets are normalized to speed up convergence and ensure all feature are equally represented in the resulting model.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

<table>
    <tr>
        <th>STEP</th>
        <th>LAYER</th>
        <th>IN. SIZE</th>
        <th>OUT. SIZE</th>
        <th>DESCRIPTION</th>
    </tr>
    <tr>
        <td>Input</td>
        <td>Input</td>
        <td>28 × 28 × 1</td>
        <td>28 × 28 × 1</td>
        <td>28 × 28 × 1 grayscale image.</td>
    </tr>
    <tr>
        <td rowspan="4">Convolution 1</td>
        <td>Convolution 7 × 7</td>
        <td>28 × 28 × 1</td>
        <td>28 × 28 × 64</td>
        <td>1 × 1 stride, SAME padding</td>
    </tr>
    <tr>
        <td>RELU</td>
        <td>28 × 28 × 64</td>
        <td>28 × 28 × 64</td>
        <td></td>
    </tr>
    <tr>
        <td>Max pooling</td>
        <td>28 × 28 × 64</td>
        <td>14 × 14 × 64</td>
        <td>2 × 2 stride</td>
    </tr>
    <tr>
        <td>Dropout</td>
        <td>14 × 14 × 64</td>
        <td>14 × 14 × 64</td>
        <td>0.6 keep rate</td>
    </tr>
    <tr>
        <td rowspan="4">Convolution 2</td>
        <td>Convolution 5 × 5</td>
        <td>14 × 14 × 64</td>
        <td>14 × 14 × 128</td>
        <td>1 × 1 stride, SAME padding</td>
    </tr>
    <tr>
        <td>RELU</td>
        <td>14 × 14 × 128</td>
        <td>14 × 14 × 128</td>
        <td></td>
    </tr>
    <tr>
        <td>Max pooling</td>
        <td>14 × 14 × 128</td>
        <td>7 × 7 × 128</td>
        <td>2 × 2 stride</td>
    </tr>
    <tr>
        <td>Dropout</td>
        <td>7 × 7 × 128</td>
        <td>7 × 7 × 128</td>
        <td>0.6 keep rate</td>
    </tr>
    <tr>
        <td rowspan="3">Convolution 3</td>
        <td>Convolution 3 × 3</td>
        <td>7 × 7 × 128</td>
        <td>5 × 5 × 256</td>
        <td>1 × 1 stride, SAME padding</td>
    </tr>
    <tr>
        <td>RELU</td>
        <td>5 × 5 × 256</td>
        <td>5 × 5 × 256</td>
        <td></td>
    </tr>
    <tr>
        <td>Dropout</td>
        <td>5 × 5 × 256</td>
        <td>5 × 5 × 256</td>
        <td>0.6 keep rate</td>
    </tr>
    <tr>
        <td>Flattening</td>
        <td>Flatten</td>
        <td>5 × 5 × 256</td>
        <td>6400 × 1</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">Fully Connected 1</td>
        <td>Fully connected</td>
        <td>6400 × 1</td>
        <td>256 × 1</td>
        <td></td>
    </tr>
    <tr>
        <td>Dropout</td>
        <td>256 × 1</td>
        <td>256 × 1</td>
        <td>0.6 keep rate</td>
    </tr>
    <tr>
        <td rowspan="2">Fully Connected 2</td>
        <td>Fully connected</td>
        <td>256 × 1</td>
        <td>128 × 1</td>
        <td></td>
    </tr>
    <tr>
        <td>Dropout</td>
        <td>256 × 1</td>
        <td>256 × 1</td>
        <td>0.6 keep rate</td>
    </tr>
    <tr>
        <td>Fully Connected 3</td>
        <td>Fully connected</td>
        <td>128 × 1</td>
        <td>43 × 1</td>
        <td></td>
    </tr>
</table>


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The first thing I added was [Xavier initializer](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf), which keeps the  scale of the gradients similar in all layers, speeding up convergence, so I don't need the params `mu` and `sigma` anymore.

Next, after taking a look at [this comparison of optimizers](http://int8.io/comparison-of-optimization-techniques-stochastic-gradient-descent-momentum-adagrad-and-adadelta/#AdaGrad_8211_experiments), I decided to use Adargad instead of Adam with a learning rate of 0.05 to speed up learning, altought probably Momentum could have been a better choice and a smaller learning rate could have achieved a greater accuracy.

I have experimented with different batch sizes and the best two values have been 196 and 256, finally keeping the later for the latest, best results.

The same applies to the number of epochs. Initially, I started with just 8 of them and, as I made changes to the architecture, params and augmented data and started getting better results, I increased it to 16, then to 32 and finally to 64 for the last few runs, which surprisingly for me decresed the accuracy in validation, but increased it in test. Anyway, after iteration 32 there are no big nor relevant gains anymore, probably because the augmented training set is not richer and variated enough.

Once I added dropout layers, I tried different values in order to prevent overfitting the training data without causing underfit and I finally found the best value for this particular case is 0.6.

Just as a reference, these are the last runs and modifications I made to the params and training set:

| EPOCHS | KEEP PROB. | BATCH SIZE | TRAIN. ACC. | VAL. ACC. | TEST ACC. | CONCLUSION / ACTION |
|--------|------------|------------|-------------|-----------|-----------|---------------------|
| 16 | 0.50 | 196 |  < 1 | 0.964 | 0.948 | Underfit. Adjust params.
| 16 | 0.50 | 256 |  < 1 | 0.977 | 0.949 | Underfit. Adjust params.
| 16 | 0.75 | 196 |  < 1 | 0.962 | 0.949 | Overfit. Adjust params.
| 16 | 0.75 | 256 |  < 1 | 0.968 | 0.949 | Overfit. Adjust params.
| 32 | 0.75 | 256 |  < 1 | 0.971 | 0.955 | Overfit. Adjust params and further augment the training set.
| 32 | 0.75 | 256 | 1.00 | 0.973 | 0.952 | Overfit. Improve augmentation algorithm.
| 32 | 0.75 | 256 | 1.00 | 0.975 | 0.954 | Overfit. Adjust keep probability.
| 32 | 0.65 | 256 | 1.00 | 0.979 | 0.958 | Overfit. Adjust keep probability.
| 32 | 0.50 | 256 | 1.00 | 0.980 | 0.956 | Underfit. Adjust keep probability.
| 32 | 0.55 | 256 | 1.00 | 0.981 | 0.960 | Underfit. Adjust keep probability.
| 32 | 0.60 | 256 | 1.00 | 0.985 | 0.961 | Looks better now. Increase epochs.
| 64 | 0.60 | 256 | 1.00 | 0.984 | 0.967 | 

Here is a plot of the accuracy of the  final architecture and params combination in training (blue) and validation (orange):

![Accuracy Plot][image6]


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results are:

* Training set accuracy of 100%
* Validation set accuracy of 98.4%
* Test set accuracy of 96.7%

I initially started with a LeNet architecture with additional convolutions and fully connected layers, as I suspected a typicall LeNet would suffer from underfit. The reason for this is that I think this approach is more interesting, as a learning experience, than reading a paper about some really good architecture and just implementing that.

The initial params for the network were just guessed and later adjusted by trial and error. Its main problem was overfit, which I addressed by adding dropout layers and augmenting the training data set.

Once I was happy with the architecture itself, I spend some time fine-tunning its params, as explained in the previous point.

Additionally, I have also calculated the precision and recall values for each sign:

|                                               SIGN |  PRECISION |     RECALL |
|----------------------------------------------------|------------|------------|
|                               Speed limit (20km/h) |  93.220339 |  91.666667 |
|                               Speed limit (30km/h) |  98.601399 |  97.916667 |
|                               Speed limit (50km/h) |  97.131682 |  99.333333 |
|                               Speed limit (60km/h) |  97.471264 |  94.222222 |
|                               Speed limit (70km/h) |  97.155689 |  98.333333 |
|                               Speed limit (80km/h) |  91.297710 |  94.920635 |
|                        End of speed limit (80km/h) | 100.000000 |  78.000000 |
|                              Speed limit (100km/h) |  96.171171 |  94.888889 |
|                              Speed limit (120km/h) |  97.111111 |  97.111111 |
|                                         No passing |  99.173554 | 100.000000 |
|       No passing for vehicles over 3.5 metric tons |  99.847328 |  99.090909 |
|              Right-of-way at the next intersection |  88.738739 |  93.809524 |
|                                      Priority road |  96.760563 |  99.565217 |
|                                              Yield |  99.722222 |  99.722222 |
|                                               Stop | 100.000000 | 100.000000 |
|                                        No vehicles |  97.674419 | 100.000000 |
|           Vehicles over 3.5 metric tons prohibited | 100.000000 |  99.333333 |
|                                           No entry | 100.000000 |  99.722222 |
|                                    General caution |  98.538012 |  86.410256 |
|                        Dangerous curve to the left |  83.333333 | 100.000000 |
|                       Dangerous curve to the right |  97.402597 |  83.333333 |
|                                       Double curve |  94.594595 |  77.777778 |
|                                         Bumpy road |  96.747967 |  99.166667 |
|                                      Slippery road |  96.710526 |  98.000000 |
|                          Road narrows on the right |  95.698925 |  98.888889 |
|                                          Road work |  98.286938 |  95.625000 |
|                                    Traffic signals |  90.909091 | 100.000000 |
|                                        Pedestrians |  71.929825 |  68.333333 |
|                                  Children crossing |  97.333333 |  97.333333 |
|                                  Bicycles crossing |  92.783505 | 100.000000 |
|                                 Beware of ice/snow |  83.108108 |  82.000000 |
|                              Wild animals crossing |  89.333333 |  99.259259 |
|                End of all speed and passing limits |  93.750000 | 100.000000 |
|                                   Turn right ahead | 100.000000 | 100.000000 |
|                                    Turn left ahead |  97.540984 |  99.166667 |
|                                         Ahead only |  99.226804 |  98.717949 |
|                               Go straight or right |  98.360656 | 100.000000 |
|                                Go straight or left |  95.238095 | 100.000000 |
|                                         Keep right |  99.698795 |  95.942029 |
|                                          Keep left | 100.000000 | 100.000000 |
|                               Roundabout mandatory |  88.043478 |  90.000000 |
|                                  End of no passing | 100.000000 |  91.666667 |
| End of no passing by vehicles over 3.5 metric tons |  95.744681 | 100.000000 |

We an see one of the worst performing signs is the "Pedestrians" one, followed by "Beware of ice/snow".

### TEST THE MODEL ON NEW IMAGES

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, plus an additional one for the reasons explained below, all with their original sizes:

<span><img src="./input/images/originals/001%20-%20Yield.jpg" alt="Yield" height="200" /></span>
<span><img src="./input/images/originals/002%20-%20Stop.jpg" alt="Stop" height="200" /></span>
<span><img src="./input/images/originals/003%20-%20Road%20Work.jpg" alt="Road Work" height="200" /></span>

<span><img src="./input/images/originals/004%20-%20Priority%20Road.jpg" alt="Priority Road" height="200" /></span>
<span><img src="./input/images/originals/005%20-%20Speed%20limit%20(30km:h).jpg" alt="Speed limit (30km/h)" height="200" /></span>
<span><img src="./input/images/originals/006%20-%20Better%20Stop.jpg" alt="Better Stop" height="200" /></span>

Once cropped and scaled down, they look like this:

![Yield][sign1] ![Stop][sign2] ![Road Work][sign3] ![Priority Road][sign4] ![Speed limit (30km:h)][sign5] ![Better Stop][sign6]

The second image might be difficult to classify because the sign is too close to the borders of the image (too much zoom), while most images from the data sets have quite generous margins around. To check if that's actually true or not in case the prediction fails, I added a 6th image of a different stop sign that is better centered in the image.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

- 👍 Yield
- 👎 Stop. Predicted Yield.
- 👍 Road Work
- 👍 Priority Road
- 👍 Speed Limit (30km/h)
- 👍 Better Stop

Wihtout considering the 6th image, the model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. However, if I had found the second stop sign first, I would have got a 100% and not notice this issue.

Depending on which case we consider, it will compare favorably or not to the accuracy on the test set of 96.7%, but neither case is representative as the size of this set is too small.

What's important to note here is that even though both stop signs are easily distingible to the naked eye, the algorithm was not able to properly identify the first one (even though it suspected it might be a stop sign, as we will see next). Therefore, the hypothesis of the first stop sign having too much zoom looks quite feasible.

Probably, further augmenting the training set and improving the augmentation algorithm to generate more varied images, as it has been suggested previously, would have help to mitigating this issue.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the "Step 3" section of the Ipython notebook.

The model is quite certain about all its predictions (which are 100% or close to it), with the exception of the incorrectly labeled Stop sign, which gets a probability of 46% of being a Yield sign, a 40% of being a Stop sign and some other residual probabilities for different Speed Limit signs, as we can see here:

![Probability Plot][image7]

The positive side of this is that the probability of a Stop sign is there and is just slightly lower than the Yield's one. That probably means that, as I already said in the previous question, with better augmented (or real) data, that problem could be mitigated and a highest accuracy in test achieved.


### (Optional) Visualizing the Neural Network

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here we can visualize the first 7 × 7 Convolutional Layer + RELU + 2 × 2 Max Pooling for the downloaded Yield, Stop and Better Stop signs:

**YIELD:**
![Yield][image8]

**STOP:**
![Stop][image9]

**BETTER STOP:**
![Better Stop][image10]

We can see that the outer shape of he sign and even the text on the Stop signs are still quite recognizable.
