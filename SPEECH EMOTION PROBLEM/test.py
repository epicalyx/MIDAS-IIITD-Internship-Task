#!/usr/bin/env python
# coding: utf-8

# In[2]:


import soundfile # to read audio file
import numpy as np
import librosa # to extract speech features
import os
import sys
import pandas as pd
from typing import Tuple
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import pickle # to save model after training
from sklearn.metrics import accuracy_score # to measure how good we are


# In[3]:


def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")

    with soundfile.SoundFile(file_name) as sound_file:
        X,sample_rate = librosa.load(file_name, sr=16000)
        
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
    return result


# In[4]:


def get_data(data_path: str,
             class_labels: Tuple = ("disgust", "fear", "happy", "neutral","sad")):
    """Extract data for training and testing.
    1. Iterate through all the folders.
    2. Read the audio files in each folder.
    3. Extract Mel frequency cepestral coefficients for each file.
    4. Generate feature vector for the audio files as required.
    Args:
        data_path (str): path to the data set folder
        flatten (bool): Boolean specifying whether to flatten the data or not.
        mfcc_len (int): Number of mfcc features to take for each frame.
        class_labels (tuple): class labels that we care about.
    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Two numpy arrays, one with mfcc and
        other with labels.
    """
    X = []
    y = []
    names = []
    cur_dir = os.getcwd()
    sys.stderr.write('curdir: %s\n' % cur_dir)
    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        sys.stderr.write("started reading folder %s\n" % directory)
        os.chdir(directory)
        for filename in os.listdir('.'):
            filepath = os.getcwd() + '/' + filename
            feature_vector = extract_feature(filepath,mfcc = True, mel = True)
            X.append(feature_vector)
            y.append(directory)
            names.append(filename)
        sys.stderr.write("ended reading folder %s\n" % directory)
        os.chdir('..')
        submission_file = pd.DataFrame(names,columns = ['File name'], index = None)
    os.chdir(cur_dir)
    return np.array(X), (y), submission_file


# In[5]:


def evaluate(test_path,saved_model_path,model_name,result_file):
    
    X_test,y_test, submission_file = get_data(data_path = test_path)
    
    if model_name == "MLP":
        model = pickle.load(open(saved_model_path, 'rb'))
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        print("Accuracy of MLP Model: {:.2f}%".format(accuracy*100))
        submission_mlp = submission_file
        submission_mlp['prediction'] = y_pred
        submission_mlp.to_csv(result_file, header=None, index=None, sep=',', mode='a')
        
    elif model_name == "CNN":
        y_test = np.array(y_test)
        lb = LabelEncoder()
        y_test = np_utils.to_categorical(lb.fit_transform(y_test))  
        x_testcnn= np.expand_dims(X_test, axis=2)
        model = load_model(saved_model_path)
        score = model.evaluate(x_testcnn, y_test, verbose=0)
        print("Accuracy of CNN Model: {:.2f}%".format(score[1]*100))
        y_pred = model.predict(x_testcnn, batch_size=32, verbose=1)
        y_preds=y_pred.argmax(axis=1)
        y_preds = lb.inverse_transform(y_preds)
        submission_csv = submission_file
        submission_cnn['prediction'] = y_preds
        submission_cnn.to_csv(result_file, header=None, index=None, sep=',', mode='a')
        


# In[22]:


print("Please enter path to test folder")
test_path = input()

print("Please enter path to saved model")
saved_model_path = input()

print("Please enter which model you want to evaluate: MLP or CNN")
model_name = input()

print("Please enter the path and filename by which you want to save result text file")
result = input()


# In[23]:


evaluate(test_path,saved_model_path,model_name,result)

