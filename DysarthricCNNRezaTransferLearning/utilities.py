import pandas as pd
import numpy as np
import os
def wav_to_index(f):  
    # To convert an utterance to a Y sample tensor
    # dictionary is pandas dataframe with the first column are true words and 
    # the second colum is a key. For example ["Zero", "D0"]
    for i,item in enumerate(dictionary.values):
        aY = np.zeros(len(dictionary))
        if "_"+item[1]+"_" in f:
            aY[i]=1
            return aY

def file_to_index(f):  
    # To convert an utterance to a Y sample tensor
    # dictionary is pandas dataframe with the first column are true words and 
    # the second colum is a key. For example ["Zero", "D0"]
    for i,item in enumerate(dictionary.values):
       
        if "_"+item[1]+"_" in f:
           return i

def index_to_word(index):
    # To convert an index to a word from the dictionary  
    # dictionary is pandas dataframe with the first column are true words and 
    # the second colum is a key. For example ["Zero", "D0"]      
    return dictionary.values[index,0]

def get_no_files_in_path(path):
    total = 0
    for root, dirs, files in os.walk(path):
        total += len(files)
    return total

def get_no_folders_in_path(path):
    total = 0
    for root, dirs, files in os.walk(path):
        total += len(dirs)

    return total

def visualize_training(training_dynamics):
        import matplotlib.pyplot as plt

        #training_dynamics=pd.read_csv(training_dynamics_path)
        loss_values = training_dynamics["TrainingLoss"]
        val_loss_values = training_dynamics["ValidationLoss"]
        epochs = range(1, len (training_dynamics['Epoch'])+1)
        plt.plot(epochs, loss_values, 'g', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        # Ploting Accuracy
        loss_values = training_dynamics["TrainingAccuracy"]
        val_loss_values = training_dynamics["ValidationAccuracy"]
        epochs = range(1, len (training_dynamics['Epoch'])+1)
        plt.plot(epochs, loss_values, 'g', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

dictionary = pd.read_csv("dictionary_UASPEECH.csv")
dd = pd.read_csv("TrainingDynamics-testing82percent.csv")
visualize_training(dd)