import numpy as np
import os
import json

#Sentences
#jsonData[#num image][0 = url, 1 = sentences][#number of sentence]
jsonValDataFile = open('merged_train.json')
jsonValDataFile = json.load(jsonValDataFile)

#Features
npyTrainlDataFile = np.load('merged_train.npy')


#both files have the same length
totalNumOfSentences = len(npyTrainlDataFile)

divisionSize = 10

smallerTrainSetSize = totalNumOfSentences / divisionSize

trainFileName = "smallerSetFolder/smaller_merged_train"
jsonExtension = "json"
npyExtension = "npy"

i = 1
totalEnd = 0
while (totalEnd < totalNumOfSentences):
	begin = ((i-1) * smallerTrainSetSize)
	end = (smallerTrainSetSize * i)
	
	smallerDataSet = jsonValDataFile[int(begin): int(end)]
	smallerFeatureSet = npyTrainlDataFile[int(begin): int(end)]
	
	#Save the file
	npyFileName = trainFileName + str(i) + "." + npyExtension
	jsonFileName = trainFileName + str(i) + "." + jsonExtension
	
	#look if there exists a folder called 'smallerSetFolder', otherwise create it
	if not os.path.exists('smallerSetFolder'):
		os.makedirs('smallerSetFolder')
	
	#first the npyFile
	np.save(npyFileName, smallerFeatureSet)
	print("Saved File: " + str(npyFileName))
	
	#now the json file
	outputFile = open(jsonFileName, 'w')
	json.dump(list(smallerDataSet), outputFile)
	print("Saved File: " + str(jsonFileName) + '\n')
	
	#increase the counters
	i += 1
	totalEnd = end