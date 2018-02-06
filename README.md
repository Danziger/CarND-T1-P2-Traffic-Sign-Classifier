CarND · T1 · P2 · Traffic Sign Classifier Project
=================================================

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="output/images/010 - Feature Map Better Stop.png" width="512" alt="Better Stop Sign's First Layer Features Map" />


Project Overview
----------------

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 


Project Structure
-----------------

The project structure has been changed a bit to have multiple files for multiple purposes instead of a single, massive Jupyter Notebook for all the code.

### `input`

Contains additional traffic sign `images` downloaded from the Internet and a CSV file with the traffic signs names, `data/signnames.csv`.

### `output/images`

Images generated for the write up.


### `src/notebooks`

Source files with the actual proposed solution, in this case in a single Jupyter Notebooks that contains all the project code, including data exploration and augmentation, neural network architecture, training and analysis, all together in a single file.


Project Evaluation
------------------

The project's writeup can be found [here](WRITEUP.md).

According to the provided guidelines, a great writeup include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as a detailed description of how to addressed each of them and the code involved (with line-number references and code snippets where necessary), images to demonstrate how the code works with examples and links to other supporting documents or external references.

The goals/steps of this project are:

* Load the data set.
* Explore, summarize and visualize the data set.
* Design, train and test a model architecture.
* Use the model to make predictions on new images.
* Analyze the softmax probabilities of the new images.
* Summarize the results with a written report.


### Dependencies

This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.


### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```


Additional Resources
--------------------

- The AI Conf 2017 - A visual and intuitive understanding of deep learning:

  https://www.youtube.com/watch?v=Oqm9vsf_hvU
  
- 3Blue1Brown Deep Learning Series:

  https://www.youtube.com/watch?v=aircAruvnKk
  
- Understand and apply CapsNet on Traffic sign classification:

  https://becominghuman.ai/understand-and-apply-capsnet-on-traffic-sign-classification-a592e2d4a4ea

