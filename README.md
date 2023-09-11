## Transfer Learning Approach: Apps For Image Classification on Dessert Problem

* Farel Firman, @farelyue
* Mikhael Adiputra, @pitydevil
* Theresia Veronika Rampisela, @theresiavr
* Tobias Ivandito Margogo Silalahi, @tobiassilalahi

## Introduction

Knowledge about nutritional intake needed every day is important for everyone to know because the fulfilment of nutritional intake each person varies depending on internal factors and external factors. Lack of knowledge about nutritional intake will cause a nutritional problem. The nutritional problem occurs because nutritional intake is not enough or excessive. It will have an impact on one’s productivity.

Dessert is food eaten after the main course and style from western cultures. Many Indonesian people now make dessert as a habit. Dessert usually is sweet courses such as fruits, ice cream, cake, etc. Dessert is used to digest the main course and acts as a mouth freshener. Furthermore, dessert also increases one’s mood. But, too many calories eaten will result in various health disorders

One can also argue that most people don't have sufficient knowledge for the ingredient in every dessert they ate. Most people use their intuition and memory to recall the ingredient in the dessert they have eaten on the ones that they will. Hence, there are probabilities that they miss an ingredient on a new particular dessert that they haven't eaten which may result in an allergic reaction.

In order to tackle those problems, our group proposed a Dessert Classification Application which can identify a dessert from a picture which can be taken by the user using their smartphone's camera and reveal the common ingredient for that particular dessert. This can help prevent people from having an allergic reaction by eating that particular dessert. Furthermore, our application can be used to display information about calories with recipes which can help the user adjust their dietary habit. Finally, our application also provides a recipe for the dessert that has been classified, hence the user is able to replicate the dessert by following the recipe on our application.


## Goals

Our goal is to create an application that can identify the type of dessert and display information regarding the number of calories contain in a dessert and, it's recipes using an image.

## Methodology

The data used is image data [Food-101](http://www.kaggle.com/kmader/food41) that comes from the Kaggle platform. This data consists of 101 types of food. Each food consists of 1000 images. From that dataset, we only choose 22 types of food as a subset, which are only dessert classes. This approach used the Convolutional Neural Network (CNN) using a pre-trained model from InceptionV3. 

The stages in image classification are:

1. Splitting data into training and validation data with a ratio of 80% and 20%.
2. Train model CNN from scratch with predetermined architecture.
3. Determine best pre-trained model by comparing performance from all three pre-trained models that are InceptionV3, VGG-16, and Resnet-50.
4. Tuning hyperparameter and used data augmentation to improve model performance.


## Results

### Splitting Data

Data is divided into two parts that are training and validation data with ratios are 80% and 20%. Training data is used to build a model and validation data is used to evaluate the model.

### CNN Model From Scratch
[CNN Model From Scratch](https://github.com/pitydevil/Bangkit-Final-Project-Dessert/blob/master/Bangkit%20Last%20Assignment.ipynb)

CNN Model is one of the main categories to do image classification tasks. CNN takes an input image and through the convolution layers used to extract features from the image and finally with activation function used to determine class from the image. The first step from building a CNN model is the determined model architecture. As for the architecture used is:

1. Convolutional Layer 2D with 32 filters each then followed by Max Pooling 2D 2x2 and Dropout is 20%. Output shape: 149x149x16
2. Convolutional Layer 2D with 64 filters each then followed by Max Pooling 2D 2x2 and Dropout is 20%. Output shape: 73x73x32
3. Convolutional Layer 2D with 128 filters each then followed by Max Pooling 2D 2x2 and Dropout is 20%. Output shape: 35x35x64
4. Fully Connected Layer 1. Output shape: 1x1x256
5. Fully Connected Layer 2. Output shape: 1x1x512
6. Output Layer. Output shape: 1x1x22

<p align = "center">
	<img height = "200" src="https://raw.githubusercontent.com/pitydevil/Bangkit-Final-Project-Dessert/master/Images/baseline%20architecture.png">
</p>

<p align = "center">
	<img width = "250" src="https://raw.githubusercontent.com/pitydevil/Bangkit-Final-Project-Dessert/master/Images/baseline%20100.png">
	<img width = "250" src="https://raw.githubusercontent.com/pitydevil/Bangkit-Final-Project-Dessert/master/Images/baseline%20300.png">
</p>

Added epoch from 100 to 300 improve accuracy model performance from the baseline model is 51.80% to 59.80%. This improvement is fairly small and tends to be stable at 50% to 60%.

### Improved CNN Model
[Improved CNN Model](https://github.com/pitydevil/Bangkit-Final-Project-Dessert/blob/master/Transfer%20Learning%20with%20Google%20Colab%20(old%20data%20split).ipynb)

The improved performance model used the Transfer Learning approach. Transfer Learning is an approach used model that has been trained before called a pre-trained model. Then, the feature from this model is fine-tuned to adjust to the case that will be implemented. There are many pre-trained models. But in this case, we only use 3 pre-trained such as InceptionV3, VGG-16, and Resnet-50 to compare the model performances. The top layer for each pre-trained model adjusts to the case Dessert problem with 22 classes. As for the architecture, we used:

1. Exclude top layer for each pre-trained model such as InceptionV3, VGG-16, and Resnet-50
2. Global Average Pooling 2D layer followed by Flatten layer used to convert featured map into a single column vector as an input to Fully Connected Layer
3. Dense Layer with 512 filters each and used ReLU activation function. Then followed by Dropout layer with 50% parameter
4. The Output layer consists of 22 classes and used the Softmax activation function

<p align = "center">
	<img height = "200" src="https://raw.githubusercontent.com/pitydevil/Bangkit-Final-Project-Dessert/master/Images/compare%20architecture.png">
</p>

<p align = "center">
	<img height = "150" src="https://raw.githubusercontent.com/pitydevil/Bangkit-Final-Project-Dessert/master/Images/compare%20result.png">
</p>

Based on the result of comparing all three pre-trained models, we found that InceptionV3 has a much better performance model compared to the others. For epoch = 5 validation accuracy is 55.09% and for epoch = 10 validation accuracy is 58.84%. Hence, InceptionV3 is chosen as the pre-trained model

The resulting model suspected of having an overfitting problem. It is seen from the accuracy in training and validation data is quite different. Overfitting problem occurs because the model is too fitted on training data and result in poor accuracy in validation data. Hence, we used data augmentation to solve this problem. Data augmentation is used to create a variation of the images that can improve the ability of the fit models to generalize.

As for the hyperparameter final model used are:

* Epoch: 140
* Batch Size :
	* Training data: 800
	* Validation data: 200
* Optimizer: Adam
* Loss Function: Categorical Cross-Entropy
* Data augmentation :
	* Rescale: 1/255
	* Rotation Range: 40
	* Width Shift Range : 0.2
	* Height Shift Range: 0.2
	* Shear Range: 0.2
	* Zoom Range: 0.2
	* Horizontal Flip: True

Based on the result from the final model we get training accuracy is 72.87% and validation accuracy is 72.52%. The results obtained are good considering the model from scratch only scored 55% accuracy and now it improved to about 17%. This model also does not have an overfitting problem, as there is not too much difference between the accuracy of training data and validation data.

