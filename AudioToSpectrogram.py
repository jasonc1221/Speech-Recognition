'''
AudioToSpectrogram will go to a user specified directory and 
convert the audio data into a spectrogram.  The data is saved
to a .png file in a folder called Spectrogram located in the same
directory as the file location of this script.
author: Data Collection member, Jonathan Evans
Date: 10 January, 2018
Note: numpy and scipy math libraries must be installed in order to
run this script 
'''

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pyaudio
import wave
import time
import os
import scipy.fftpack
import struct

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2 
RATE = 3000

#np.set_printoptions(threshold=np.nan)

def getAudioData(directory):
    #open the wave file
    wf = wave.open(directory, 'rb')
    #get the packed bytes
    raw_sig = wf.readframes(wf.getnframes())
    #convert the packed bytes into integers
    audio_sig = np.fromstring(raw_sig, 'Int16')
    wf.close()#close the file
    return audio_sig
    
def saveSpectrogramData(data):
    '''
    #get the file name without the .wav extenstion
    tempLex = fileName[:-7]#removes (#).wav
    newFileName = ''.join(tempLex)								#joins the char array into a string
    #get the full directory for the location of the png file
    directory = path + newFileName + ".png"
    '''
    #get the spectrogram figure and save it
    f, t, Sxx = signal.spectrogram(data, RATE)
    '''
    #To save spectrogram into a .png file
    fig = plt.figure(frameon = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.xlim(0,3)
    plt.savefig(directory, bbox_inches='tight')
    plt.close('all')
    '''
    return Sxx
    
'''
path = input("Enter the FULL path to the folder in which all of the audio files are located." + "\n"
             + "Make sure that there are only .wav files in this folder: ")
current_directory = os.getcwd()
spec_path = os.path.join(current_directory, 'Spectrogram\\')
if not os.path.exists(spec_path):
    #create the Spectrogram folder if it doesn't already exist
    os.makedirs(spec_path)

#get all of the files in the path that is entered by the user
listing = os.listdir(path)

for fileName in listing:
    print ("current file is: " + fileName)
    directory = path + fileName 
    audio_sig = getAudioData(directory)
    saveSpectrogramData(audio_sig, fileName, spec_path)
    print("Spectrogram for " + fileName + " is saved")

'''
    
