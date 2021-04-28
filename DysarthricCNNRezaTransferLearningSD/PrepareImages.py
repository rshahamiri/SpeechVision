#!/usr/bin/env python
# coding: utf-8

# In[1]:


import subprocess
from os import path
import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import wave
import pylab
import pyprog
import utilities

SETTINGS_DIR = os.path.dirname(os.path.realpath('__file__'))


# In[2]:


# prepare folder sturctures
# Crteate all directories based on lables
def create_directories(directories_list, base_path="images/Train/Control/"):   
    for item in directories_list:
        if not os.path.exists(base_path+item):
            os.makedirs(base_path+item)


# In[3]:


def organise_waves_in_directories(waves_path, new_path ="images/Train/Control/" ):
    # Move all wave files to thier labeld folder. 
    # For example, all D0 wave samples are put in D0 folder
    # NOTE: THE ORIGINAL DATA WILL BE MOVED FROM THE ORIGINAL PATH
    for directory, s, files in os.walk(waves_path):
        for f in files:         
            file_path=directory+"/"+f
            if ("wav" in f):
                for label in dictionary.iloc[:,1]:
                    if "_"+label+"_" in f:
                        print ("Moving",f)
                        os.rename(file_path, new_path + label +"/"+f)
                        break


# In[4]:


def save_a_spectrogram(wav_file_path, file_name):
    # This function converts a wave file to a spectrogram image
    sound_info, frame_rate = get_wav_info(wav_file_path)
    pylab.figure(num=None, figsize=(8, 6))
    pylab.subplot(111)
    pylab.axis('off')
    pylab.subplots_adjust(left=0,right=1,bottom=0,top=1)
    #pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(wav_file_path.split(".wav")[0]+'.jpg', format='jpg')
    pylab.close()


# In[5]:


def get_wav_info(wav_file):
    # This function is used by save_spectrogram to read wave files
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


# In[6]:


def create_spectrogrms(waves_path):
    # This function creates spectorgarms of wave files in the same folder
    # WARNING: THE ORIGIANL WAVE FILE WILL BE DELETED
    inp = input("All wave files in "+waves_path+" will be deleted. Do you want to procees? y/n")
    if (inp=="y"):
        prog = pyprog.ProgressBar("Creating spectrograms: ", " Done",
                                  utilities.get_no_files_in_path(waves_path))
        # Show the initial status
        prog.update()
        no_processed=0
        
        for directory, s, files in os.walk(waves_path):
            for f in files:
                
                prog.set_stat(no_processed)
                prog.update()
                
                file_path=directory+"/"+f
                if ("wav" in f):
                    save_a_spectrogram(file_path,f)
                    os.remove(file_path)
                no_processed +=1
        prog.end()


# In[47]:


#Local Database
database_path=SETTINGS_DIR+"/UASPEECH/Dysarthric/Test/M01/"
dictionary = pd.read_csv("dictionary_UASPEECH.csv")
base_path = SETTINGS_DIR+ "/images/Dysarthric/Test/M01/"


# In[48]:


# Create labeled directorirs       
create_directories(dictionary.iloc[:,1] , base_path=base_path)


# In[49]:


# Detele _UW folders
import os, shutil
for directory, s, files in os.walk(base_path):
        #for f in files:
            #print (folder)
            if ("_UW" in directory):
                shutil.rmtree(directory)
                print ("Deleted", directory)


# In[50]:


# Organise waves into labled directories
organise_waves_in_directories(database_path, new_path =base_path)  


# In[51]:


# Replace waves with spectograms
create_spectrogrms(base_path)


# In[ ]:


# To delete a folder
import shutil

#shutil.rmtree(SETTINGS_DIR+ "/UASPEECH/Test/Combined")


# In[ ]:





# In[ ]:




