
## Writeup for P2 (Traffic Sign Recognition)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is : **34799**
* The size of the validation set is : **4410**
* The size of test set is : **12630**
* The shape of a traffic sign image is :**(34799, 32, 32, 3)**
* The number of unique classes/labels in the data set is : **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It includes the histogram representation of the counts of the images in each class and a sample image of each type

<figure>
    <img src="https://github.com/NRCar/P2/blob/master/saved/original_set_histogram.png" height="270" width="480" />
    <figcaption text-align: center>Original Training Set Histogram - Some classes have less samples than others</figcaption>
</figure>


<figure>
    <img src="https://github.com/NRCar/P2/blob/master/saved/samples.png" />
    <figcaption>Original Training Set Samples from each Class</figcaption>
</figure>

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, i decided to convert all the images to gray scale since the the color information was not very reliable as it appeared to be different in different lighting and seems to worsen the training accuracy
I also normalized the images so that the pixel values were close to zero making training much easier on the set since the values were distributed wround zero with a close to zero mean.

Here is an example of a traffic sign images after processing with the above steps vs the original images sample above.

<figure>
    <img src="https://github.com/NRCar/P2/blob/master/saved/processed_images.png" />
    <figcaption>Processed training data</figcaption>
</figure>


I decided to generate additional data because there were many classes without enough samples to get the model properly trained. And also within the classes that had a higher number of samples the samples did not seem to have enough varying characteristics to make the training model robust.

To add more data to the the data set, I used the following techniques based on the paper by Pierre Sermanet and Yann LeCun
 * Randomly rotate the images between -15 and +15 degrees
 * Skew the images in position by +/- 2 pixels in the horizontal or vertical direction
 * Random zoom between 0.8 and 1.2 times 

This improved the sample size of the trainign set

<figure>
    <img src="https://github.com/NRCar/P2/blob/master/saved/augumented_set_histogram.png" />
    <figcaption>Augumented set Histogram</figcaption>
</figure>

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x64	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flattening	      	| output 400 				|
| Fully connected		| Output 120  |
| RELU					|												|
| Dropout					|			keep_prob = 0.7									|
| Fully connected		| Output 84  |
| RELU					|												|
| Dropout					|			keep_prob = 0.7									|
| Fully connected		| Output 43  |
| Softmax				|         									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the adam optimizer with a batch size of 200 with 15 epochs and a learning rate of 0.001 since this was the largest batch i could fit in memory, and any further epochs or reduced learning rate did not give better results in the trainign model

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of : **1.0**
* validation set accuracy of : **0.972** 
* test set accuracy of : **0.962**

* What was the first architecture that was tried and why was it chosen?
A) The first approcach was to just use the Lenel architechture that was used for the  MNIST as the basis which lead to an accuracy of 0.89 of the validation set
* What were some problems with the initial architecture?
A) The initial architechture after the processed images barely gave the 0.93 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
A)  Adding the dropout between the fully connected layers seemed to make the model more resilient.

* Which parameters were tuned? How were they adjusted and why?
A) I tried to adjust the learning rate, the number of epochs and the batch size andplaying around and with trial and error i arrived at the above values
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
A) The dropout layer helped make the model resilent

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<figure>
    <img src="https://github.com/NRCar/P2/blob/master/saved/internet_images.png" />
    <figcaption>All the Internet images</figcaption>
</figure>

I assumed the first stop sign image to be difficult to classify since it was skewed and taken from an angle

I then processed the images by scaling them to 32 x 32 and grayscaling and normalizign as with the other sets

<figure>
    <img src="https://github.com/NRCar/P2/blob/master/saved/processed_internet_images.png" />
    <figcaption>Processed internel images</figcaption>
</figure>

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

<figure>
    <img src="https://github.com/NRCar/P2/blob/master/saved/predictions_internet_images.png" />
    <figcaption>Prediction of the internel images</figcaption>
</figure>


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The prediction probabilities can be visualized using the bar graphs in the image below ( generated by the code)

<figure>
    <img src="https://github.com/NRCar/P2/blob/master/saved/probabilities_internet_images.png" />
    <figcaption>Prediction probabilities the internel images</figcaption>
</figure>



