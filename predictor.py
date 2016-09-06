import os
import numpy as np 
import pandas as pd
from scipy.io import loadmat
import shutil
from functools import partial
from multiprocessing import Pool
           

def ieegGetFilePaths(directory, extension='.mat'):
	filenames = sorted(os.listdir(directory))
	files_with_extension = [directory + '/' + f for f in filenames if f.endswith(extension) and not f.startswith('.')]
	return files_with_extension


def ieegProcessMatFiles(raw_file_path):    
	d = loadmat(raw_file_path, squeeze_me=True)
	
	x = d['dataStruct']
	ieegData=x['data']
    	channelIndices = x['channelIndices']	
    	sequence = x['sequence']
	iEEGsamplingRate = x['iEEGsamplingRate']
	
	print '---------------' + os.path.basename (raw_file_path)+ '-------- START' + 'sequence:' + str(sequence)           
	print 'ieegData:' + str(ieegData)  
	print 'channelIndices:' + str(channelIndices)
	print 'iEEGsamplingRate:' + str(iEEGsamplingRate)
	print '----------------------- END ' + 'sequence:' + str(sequence)  

if __name__=="__main__":	
	rawMatFiles= ieegGetFilePaths("train_1")

	for singleRawMatFile in rawMatFiles:
    		ieegProcessMatFiles(singleRawMatFile)	
    		
	
