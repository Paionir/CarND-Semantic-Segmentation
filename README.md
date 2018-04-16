# Udacity Self-driving Car Nanodegree
# Term 3, project 2
==================================================
# Semantic Segmentation

### Introduction
In this project, I had to label the pixels of a road in images using a Fully Convolutional Network (FCN).

#### Frameworks and Packages 

 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

##### Dataset
Kitti Road dataset
(http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  

##### Implementation
The implementation had to be written inthe `main.py` module indicated by the "TODO" comments.  


### General approach

The main goal of this project is to create and train a FCN starting from an already trained model (VGG16) of a CNN.
The first step taken was to transform the CNN in a FCN using 1x1 convolutional layer instead of the dense fully connected layer in order to preserve the spatial information of the input image. This step is impelented in the functions 
```python 
load_vgg(sess, vgg_path)
``` 
and 
```python
layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
```

The first function loads the VGG16 pretrained model with its weights, the second function adds to this model the "decoding" layers (as in the FCN terminology the VGG16 model can be seen as an encoder).

The decoding layers are 1x1 convolutions and transpose convolution layers to "upsample" the image to its original size. The decoder also uses the skip method to get informations from other layers of the encoder network.

In order to avoid overfitting, I used an L2 regularization for each convolutional and transpose convolutional layer, and a dropout probability of 0.2.  

The kernel where initialized with a normal distribution.  

The number of epochs to train was determined only according to an economic evaluation (I had just 2 free hours left of GPU time to spend on Floydhub)
The batch size was set so that the training used around 60% of total GPU memory ( to avoid the "exhausted resources" error)


#### Parameters

The parameters used are the following:

	*	EPOCHS = 45
	*   BATCH SIZE = 5
	*	KEEP PROB = 0.8
	*   LEARNING RATE = 1e-3
	*	L2 SCALE = 1e-3
	*	STD DEVIATION KERNEL INITIALIZER = 1e-2

 

##### Run

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
