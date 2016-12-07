import gensim
import numpy
import logging
import keras
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
#from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.preprocessing import LabelEncoder
#from sklearn.pipeline import Pipeline
from itertools import zip_longest, count
import json 
import nltk
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape, RepeatVector, Masking
from keras.layers.recurrent import LSTM,GRU
from collections import Counter
from keras.layers.wrappers import TimeDistributed


# Load Files into Memory.

image_features = numpy.load('smaller_merged_train.npy')
with open ('smaller_merged_train.json') as f:
    image_captions = json.load(f)

# Remove this to end fixed testing with 5 data features
image_features = image_features[:5]


# Generates a list containing all sentences appended with start and termination symbols.
# select_list contains the selector indexed sentence.

input_list = []
select_list = []
selector = 0
for i in range(len(image_captions)):
    sent_temp = []
    for j in range(len(image_captions[i][1])):
        temp = ['<s>']
        temp.extend(image_captions[i][1][j])
        temp.append("</s>")        
        input_list.append(temp)
        if (j==selector):
            select_list.append(temp)

# Remove this to end fixed testing with 5 sentences
select_list = select_list[:5]

# Generation of nb_samples parameter 

nb_samples = len(select_list)

# Generation of Dictionary and vocab_size parameter

word_dict = []

for i in range(len(select_list)):
    for j in range(len(select_list[i])):
        if (select_list[i][j] not in word_dict):
            word_dict.append(select_list[i][j])

# Vocab size increased by 1 to avoid 0 as an integer label.

vocab_size = len(word_dict) + 1

print("Size of this Vocabulary is %r" %vocab_size) 

#Generation of MAXLEN Parameter            

MAXLEN = 0

for i in range(len(select_list)):
    if (len(select_list[i])>MAXLEN):
        MAXLEN = len(select_list[i])

print("Maxlen for this Data is %r "%MAXLEN)       

char_labels = {ch:i for i, ch in enumerate(word_dict)}

labels_char = {i:ch for i, ch in enumerate(word_dict)}

# Generate Input and Output Streams

x_train = []
y_train = []

for i in range(len(select_list)):
    x_train.append(select_list[i][:-1])
    y_train.append(select_list[i][1:])

x_train = numpy.asarray(x_train)


#Convert Streams to their Integer valued counterparts

for i in range(len(x_train)):
    for j in range(len(x_train[i])):
        x_train[i][j] = char_labels[x_train[i][j]] + 1
        
x_train = numpy.asarray(x_train)


x_train = sequence.pad_sequences(x_train,maxlen = MAXLEN, padding = 'post')

        
for i in range(len(y_train)):
    for j in range(len(y_train[i])):
        y_train[i][j] = char_labels[y_train[i][j]]
        
y_train = sequence.pad_sequences(y_train,maxlen = MAXLEN, padding = 'post')

# Convert Outputs to One-Hot form

y_out = numpy.zeros((nb_samples,MAXLEN,vocab_size))

for i in range(len(y_train)):
    for j in range(len(y_train[i])):
        y_out[i][j][y_train[i][j]] = 1

# Model Generaton for GRU based system
y_train = y_out

num_hidden_units_mlp = 1024
image_model = Sequential()
image_model.add(Dense(128,input_dim = 4096))

language_model = Sequential()
language_model.add(Embedding(vocab_size, 256, input_length=MAXLEN))
language_model.add(GRU(output_dim=128, return_sequences=True))
language_model.add(TimeDistributed(Dense(128)))

image_model.add(RepeatVector(MAXLEN))

model = Sequential()
model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))

model.add(GRU(256, return_sequences=True))

model.add(TimeDistributed(Dense(vocab_size)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')    

# Keep best weights and save callback history

# define the checkpoint

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit([image_features, x_train], y_train, batch_size=1, nb_epoch=40)




