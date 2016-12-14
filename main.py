import numpy
import copy
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import json
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape, RepeatVector, Masking
from keras.layers.recurrent import LSTM,GRU
from keras.layers.wrappers import TimeDistributed


# Load Files into Memory.

image_features = numpy.load('merged_train.npy')
with open('merged_train.json') as f:
    image_captions = json.load(f)

# Remove this to end fixed testing with 5 data features

# image_features = image_features[:5]

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
        if j == selector:
            select_list.append(temp)

# Remove this to end fixed testing with 5 sentences
# select_list = select_list[:5]

# Generation of nb_samples parameter

nb_samples = len(select_list)

# Generation of Dictionary and vocab_size parameter

word_dict = []


for i in range(len(select_list)):
    for j in range(len(select_list[i])):
        if select_list[i][j] not in word_dict:
            word_dict.append(select_list[i][j])

# Vocab size increased by 1 to avoid 0 as an integer label.

vocab_size = len(word_dict) + 1

print("Size of this Vocabulary is %r" % vocab_size)

# Generation of MAXLEN Parameter

MAXLEN = 0

for i in range(len(select_list)):
    if len(select_list[i])>MAXLEN:
        MAXLEN = len(select_list[i])
MAXLEN-=1
print("Maxlen for this Data is %r "%MAXLEN)

char_labels = {ch:i for i, ch in enumerate(word_dict)}
labels_char = {i:ch for i, ch in enumerate(word_dict)}

# Convert Streams to their Integer valued counterparts
caption_list = copy.deepcopy(select_list)

for i in range(len(caption_list)):
    for j in range(len(caption_list[i])):
        caption_list[i][j] = char_labels[caption_list[i][j]] + 1


caption_list = sequence.pad_sequences(caption_list, maxlen=MAXLEN+1, padding='post')
x_train = caption_list[:, :-1]
y_train = caption_list[:, 1:, numpy.newaxis]


# CNN-GRU Model

image_model = Sequential()
image_model.add(Dense(128,input_dim=4096))
image_model.add(RepeatVector(MAXLEN))

language_model = Sequential()
language_model.add(Embedding(vocab_size, 256, input_length=MAXLEN,mask_zero=True))

model = Sequential()
model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))

model.add(GRU(256, return_sequences=True))

model.add(TimeDistributed(Dense(vocab_size)))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

print(x_train.shape)
print(y_train.shape)
model.fit([image_features, x_train], y_train, batch_size=200, nb_epoch=20)

# Serialize model to JSON

model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)

# Save weights

model.save_weights("model.h5")
print ("Model Saved")


# Model Evaluation

# scores = model.evaluate([image_features, x_train], y_train,verbose = 0)
# print ("%s: %.2f%%" %(model.metrics_names[1],scores[1]*100))

# Prediction Routine

# x_test = x_train[:1]
# images_test = image_features[:1]
#
# x_test = numpy.asarray(x_test)
#
# images_test = numpy.asarray(images_test)
# prediction = model.predict_classes([images_test,x_test])
#
# print(prediction)
