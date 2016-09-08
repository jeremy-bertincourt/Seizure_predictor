import re
import os
import shutil
import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
from functools import partial
from multiprocessing import Pool
from sklearn.preprocessing import normalize
from scipy.signal import correlate, resample
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
       
 
def getFiles(directory, extension='.mat'):
   	# Sort all the files of the directory
	filenames = sorted(os.listdir(directory))
 
    	# Obtain all the files of the directory
	files_with_extension = [directory + '/' + f for f in filenames if f.endswith(extension) and not f.startswith('.')]
 
	# Return all the files of the directory
    	return files_with_extension
 
def verifyElectrodeValue(X, y):
	clf = RandomForestClassifier(n_estimators = 100)
 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
 
	clf.fit(X_train, y_train)
	predictions = clf.predict_proba(X_test)[:, 1]
                       
	result = roc_auc_score(y_test, predictions)
 
	return result
           
def processMatFile(filePath, mode, element=0, electrodeIndices=0):
	# Load the file   
	try:
		d = loadmat(filePath)
	except ValueError as ex:
		print(u'Error loading MAT file {}: {}'.format(os.path.basename (filePath), str(ex)))
 
	# Extract each data from the file
 	datastruct = d['dataStruct']
	sequence = datastruct['sequence'][0][0][0][0]
	channelIndices = datastruct['channelIndices'][0][0][0]
	iEEGsamplingRate = datastruct['iEEGsamplingRate'][0][0][0][0]
	nSamplesSegment = datastruct['nSamplesSegment'][0][0][0][0]
	ieegData = datastruct['data'][0][0].transpose()
 
	# extract the class of the file
	header = re.split(r"[_.]", os.path.basename (filePath))
	category = header[2]
 
	'''
	for i in range(16):
		plt.subplot(8, 2, i + 1)
 		plt.plot(ieegData[i])
           
	plt.show()
           
	X = None
	correlations = None
	for i in range(16):
		correlations_i = np.array([])
                        for j in range (16):
                                   if i != j:
                                               corr_i = correlate(ieegData[i], ieegData[j], mode='same')
                                               correlations_i = np.concatenate([correlations_i, corr_i])
                   
                                   if correlations is None:
                                               correlations = correlations_i
                                   else:
                                               correlations = np.vstack([correlations, correlations_i])
 
                        ieegData = np.column_stack([ieegData, correlations])
       
                        X = np.vstack([X, ieegData]) if X is not None else ieegData
	'''
	data = []
	if mode == 'init':
		y = np.repeat(category, 1, axis=0)
		data.append(ieegData[element].transpose())
	else:
 		y = np.repeat(category, len(electrodeIndices), axis=0)
		for alpha in electrodeIndices:
			data.append(ieegData[alpha].transpose())
 
	#print 'y_shape = ', y.shape
	#print 'X_shape = ', data.shape
 
 	# Compute the FFT and normalize the result
	data = np.fft.fft(data)
	data = normalize(data, copy=False)
 
	y = map(int, y)
 
	# Return the data and the class
	return data, y
 
def chooseBestElectrodes(TestFiles, threshold):
	data = None
	y = None
	bestElectrodeIndices = []
 
	# Obtain best electrodes
	for i in range(16):
		for TestFile in TestFiles:
			subData, category = processMatFile(TestFile, 'init', i)
			if y is None:
				y = category
			else:
				y = np.concatenate((y, category), axis=0)
			if data is None:
				data = subData
			else:
				data = np.concatenate((data, subData), axis=0)
 
		#print 'y_shape = ', y.shape
		#print 'X_shape = ', data.shape
 
                # Compute logistic regression on each electrode to determine their value in the data set
		result = verifyElectrodeValue(data, y)
 
		# Keep only the electrodes that do better than chance
		if result > threshold:
			bestElectrodeIndices.append(i)
			#print result
 
	return bestElectrodeIndices
 
def processData(MatFiles, electrodeIndices):
	data = None
	y = None
 
	# Obtain data
	for MatFile in MatFiles:
		subData, category = processMatFile(MatFile, 'normal', electrodeIndices=electrodeIndices)
		if y is None:
			y = category
		else:
			y = np.concatenate((y, category), axis=0)
		if data is None:
			data = subData
		else:
			data = np.concatenate((data, subData), axis=0)
 
	#print 'y_shape = ', y
	#print 'X_shape = ', data.shape
 
	result = verifyElectrodeValue(data, y)
 
	# Convert dictionnary of data to a dataFrame
	#new_dataFrame = pd.DataFrame(dataDict.items(), columns= ['DataNumber','Data'])
 
	# return array of data and class
	#return new_dataFrame['Data'], y
 
	return result
 
if __name__=="__main__":
 
	initMatFiles= getFiles("test")
	listThreshold = [0.4, 0.5, 0.6, 0.7]
	for i in listThreshold:
		electrodeIndices = chooseBestElectrodes(initMatFiles, i)
 
		print electrodeIndices
 
		MatFiles= getFiles("test")
		globalResult = processData(MatFiles, electrodeIndices)
 
		print globalResult
                        
                        
