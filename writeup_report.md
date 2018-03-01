**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/Figure_1.png "Model Visualization"
[image2]: ./output/center_2018_01_04_22_16_55_583.jpg "Center"
[image3]: ./output/left_2018_01_04_22_18_59_389.jpg "Left"
[image4]: ./output/right_2018_01_04_22_19_00_965.jpg "Right"


My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run1.mp4 a visualization video
* writeup_report.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python3 drive.py model.h5 run1
```

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train the model to drive in proper straight and recognize turnings.

```
My first step was to use a convolution neural network model similar to the nvidia end-to-end convolution network. The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. My model consists of a convolution neural network with 5 neural layers with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 117-137). The model includes RELU layers to introduce nonlinearity (code line 120-128), and the data is normalized in the model using a Keras lambda layer (code line 118). 

The model contains dropout layers in order to reduce overfitting (model.py lines 91-95). I set the epoches as 2 (model.py code line 95). As I increase the number of epoches and batch size, I need to contain dropout layer in order to reduce overfitting. The model was trained and validated on different data sets to ensure that the model was not overfitting. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set at a ratio of 9/1. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that the 'relu' activation layers are used on the convolution layers as well as dropout layers in between fullu converted layers. After that the overfitting problem is fixed, as can see from the 'Figure_1.png' included in the submission that both the training and validation sets follows the same trend across the training.
```
![alt text][image1]

Then I preprocessed the data as introduced in the project.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 137). Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. And combinations of swinging between lines, reversed direction and from the sample data.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track because of the sharp turnings. To improve the driving behavior in these cases, I adjusted the wheel angel so that the model will learn to turn more angle when it encounters these scenarios.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 117-137) consisted of a convolution neural network with the following layers and layer sizes:

```
Layer (type) Output Shape Param # Connected to

Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3))

Cropping2D(cropping=((70, 25), (0, 0)))

Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu')

Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu')

Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu')

Convolution2D(64, 3, 3, activation='relu')

Convolution2D(64, 3, 3, activation='relu')

```

Added dropout to avoid overfitting:

```
model.fit_generator(train_generator, samples_per_epoch =
    len(train_set), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_set), 
    nb_epoch=2, verbose=1)

```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recognize the edge. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles.

After the collection process, I had 16000 number of data points. I then preprocessed this data by 


I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 25 as evidenced by training accuracy. I used an adam optimizer so that manually training the learning rate wasn't necessary.
