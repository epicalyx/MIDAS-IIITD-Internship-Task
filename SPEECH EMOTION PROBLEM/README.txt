CONTEXT

The aim here is to develop an automatic speech emotion classification model. 
Emotion is an important aspect of communication, but is really challenging to be modelled using automatic techniques. Unlike our visual cortex system, our ability to understand each
other isn't that precise. Humans tend to misunderstand each other, and this poses a big challenge in data annotaion itself, for speech emotion classification purposes. 

Even if this is resolved, next comes the issue with the features to be extracted. Usually hand crafted features are chosen for such taks but they require domain knowledge. 
Two main sets of features for speech problems are: Acoustic Features and Contextual Features. 
Acoustic Features consists of parameters like pitch, tone, volume etc, while contextual features represent the arousal and valence of speech.

Here, acoustic features are being used. 


DATA SOURCE

The dataset taken here is "MELD - A Multimodal Multi-Party Dataset"
It can be found here: https://github.com/SenticNet/MELD

It contains audio files and has 5 emotion labels, namely, DISGUST, FEAR, HAPPY, NEUTRAL and SAD.


DIRECTORY STRUCTURE

SPEECH EMOTION PROBLEM
    .
    +-- Models/
    |   +-- MLP	(stores pre-trained model for multi-layer perceptron)
	|	+-- CNN (stores pre-trained model for CNN)         
    +-- Notebook	(contains code notebook)
    +-- SubmissionFiles	(contains submission files for MLP and CNN model)  	


SETUP

To be able to run the code and reproduce the results, please follow these steps:

Option A
1. Clone this repository. Run this command in your prompt/shell:
- git clone https://github.com/epicalyx/MIDAS-IIITD-Internship-Task.git

2. Create an environment using "environment_ser.yml" file. Run this command:
- conda env create -f environment_ser.yml

Now, an environment will be created and all the dependencies will be loaded.

3. Activate it by running:
- conda activate sep

4. To test the model, please make sure all the weight files and test data are present in your system and then run:
- python test.py

Option B
1. To clone the repository, create an environment and setup project dependencies run:
`$ ./setup.sh` 
2. To test the model, please make sure all the weight files and test data are present in your system and then run:
`$ python test.py`

I had some issues in running "conda" commands in bash script files in my Windows OS, though they might work out for you. I have therefore provided an alternate as "environment_ser.yml"
file in option A, that does the same things and worked fine in my system :)


METHODOLOGY

Feature Extraction:
Mainly two features have been worked upon here:
1. Mel-Spectogram Frequency, which are representations of the short-term power spectrum of a sound. 
2. MFCC (Mel-frequency cepstral coefficients); these coefficients collectively make up MFC.
They have been extracted using Librosa-a library to analyse audio files.

Models:
Two models have been proposed here:

1. Multi Layer Perceptron- Since we are given with a complex classification task, the first choice would be a simple Artificial Neural Network.
The model discussed here is a dense neural network, having 300 units in a hidden layer, and tuned to experiment with different architectures.
It was then evaluated on the validation set provided.

MLP Pipeline:
- Converting training and validation data in the form of feature vectors and labels
- Setting hyperparametrs for MLP
- Training the model
- Prediction and evaluation
- Saving the model for future deployment

2. Convolutional Neural Network- CNNs have shown promising results for feature extraction in iamge and audio classification problems. A 6 layer CNN has been proposed with
1 dimensional convolutional layers, a dropout and a max pooling layer.
Classification is done via a final output layer using softmax activation function. The feature extraction layers use ReLU as activation function.

CNN Pipeline:
- Preparing feature vectors and labels to feed and evaluation purposes.
- Building model
- Traning the model
- Prediction and evaluation
- Saving the model for future deployment


PERFORMANCE

Model     Accuracy
MLP        55.66%
CNN        62.29%












