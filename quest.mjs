export const question = () => {
  return {
    1: `pip install neural-style-transfer
        from neuralstyletransfer.style_transfer import NeuralStyleTransfer
        from PIL import Image
        content_url = 'https://i.ibb.co/6mVpxGW/content.png'
        style_url = 'https://i.ibb.co/30nz9Lc/style.jpg'
        nst = NeuralStyleTransfer()
        nst.LoadContentImage(content_url, pathType='url')
        nst.LoadStyleImage(style_url, pathType='url')
        output = nst.apply(contentWeight=1000, styleWeight=0.01, epochs=20)
        output.save('output.jpg')`,

    2: `import numpy as np
        import pandas as pd
        
        data=pd.read_csv('HR_comma_sep.csv')
        data.head()
        from sklearn import preprocessing

        le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
data['salary']=le.fit_transform(data['salary'])
data['Departments ']=le.fit_transform(data['Departments '])
Split the dataset
# Spliting data into Feature and
X=data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours',
'time_spend_company', 'Work_accident', 'promotion_last_5years', 'Departments ', 'salary']]
y=data['left']
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 70%
training and 30% test
Build Classification Model
# Import MLPClassifer
from sklearn.neural_network import MLPClassifier
# Create model object
clf = MLPClassifier(hidden_layer_sizes=(6,5),
random_state=5,
verbose=True,
learning_rate_init=0.01)


clf.fit(X_train,y_train)
Make Prediction and Evaluate the Model
# Make prediction on test dataset
ypred=clf.predict(X_test)
# Import accuracy score
from sklearn.metrics import accuracy_score
# Calcuate accuracy
accuracy_score(y_test,ypred)

        `,

    3: `
       TrainingImagePath='/Users/farukh/Python Case Studies/Face Images/Final Training Images'
from keras.preprocessing.image import ImageDataGenerator
# Understand more about ImageDataGenerator at below link
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# Defining pre-processing transformations on raw images of training data
# These hyper parameters helps to generate slightly twisted versions
# of the original image, which leads to a better model, since it learns
# on the good and bad mix of images
train_datagen = ImageDataGenerator(
shear_range=0.1,
zoom_range=0.1,
horizontal_flip=True)
# Defining pre-processing transformations on raw images of testing data
# No transformations are done on the testing images
test_datagen = ImageDataGenerator()
# Generating the Training Data
training_set = train_datagen.flow_from_directory(
TrainingImagePath,
target_size=(64, 64),
batch_size=32,
class_mode='categorical')

# Generating the Testing Data
test_set = test_datagen.flow_from_directory(
TrainingImagePath,
target_size=(64, 64),
batch_size=32,

class_mode='categorical')
# Printing class labels for each face
test_set.class_indices

TrainClasses=training_set.class_indices
# Storing the face and the numeric tag for future reference
ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
ResultMap[faceValue]=faceName
# Saving the face map for future reference
import pickle
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
pickle.dump(ResultMap, fileWriteStream)
# The model will give answer as a numeric tag
# This mapping will help to get the corresponding face name for it
print("Mapping of Face and its ID",ResultMap)

# The number of neurons for the output layer is equal to the number of faces
OutputNeurons=len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)




from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
'''Initializing the Convolutional Neural Network'''
classifier= Sequential()




classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activ
ation='relu'))


classifier.add(MaxPool2D(pool_size=(2,2)))
'''############## ADDITIONAL LAYER of CONVOLUTION for better accuracy #########
########'''
classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))


classifier.add(Flatten())
'''# STEP--4 Fully Connected Neural Network'''
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(OutputNeurons, activation='softmax'))
'''# Compiling the CNN'''
#classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])
###########################################################
import time
# Measuring the time taken by the model to train
StartTime=time.time()
# Starting the model training
classifier.fit_generator(
training_set,
steps_per_epoch=30,
epochs=10,
validation_data=test_set,
validation_steps=10)
EndTime=time.time()
       


import numpy as np
from keras.preprocessing import image

ImagePath='/content/drive/MyDrive/Deep Learning Lab/Face-
Images/Face Images/Final Testing Images/face11/1face11.jpg'

test_image=image.load_img(ImagePath,target_size=(64, 64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image,verbose=0)
#print(training_set.class_indices)
print('####'*10)
print('Prediction is: ',ResultMap[np.argmax(result)])
       `,

    4: `import torch
       import torch.nn as nn
       import torch.nn.functional as F
       import torch.optim as optim
       torch.manual_seed(1)
       lstm = nn.LSTM(3, 3) # Input dim is 3, output dim is 3
       inputs = [torch.randn(1, 3) for _ in range(5)] # make a sequence of length 5
       # initialize the hidden state.
       hidden = (torch.randn(1, 1, 3),
       torch.randn(1, 1, 3))
       for i in inputs:
       # Step through the sequence one element at a time.
       # after each step, hidden contains the hidden state.
       out, hidden = lstm(i.view(1, 1, -1), hidden)
       inputs = torch.cat(inputs).view(len(inputs), 1, -1)
       hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3)) # clean out hidden state
       out, hidden = lstm(inputs, hidden)
       print(out)
       print(hidden)
       Output:
       tensor([[[-0.0187, 0.1713, -0.2944]],
       [[-0.3521, 0.1026, -0.2971]],
       [[-0.3191, 0.0781, -0.1957]],
       [[-0.1634, 0.0941, -0.1637]],
       [[-0.3368, 0.0959, -0.0538]]], grad_fn=<StackBackward0>)
       (tensor([[[-0.3368, 0.0959, -0.0538]]], grad_fn=<StackBackward0>), tensor([[[-0.9825, 0.4715,
       -0.0633]]], grad_fn=<StackBackward0>))
       Prepare Data:
       def prepare_sequence(seq, to_ix):
       idxs = [to_ix[w] for w in seq]
       return torch.tensor(idxs, dtype=torch.long)
       
       training_data = [
       
       # Tags are: DET - determiner; NN - noun; V - verb
       # For example, the word "The" is a determiner
       ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
       ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
       ]
       word_to_ix = {}
       # For each words-list (sentence) and tags-list in each tuple of training_data
       for sent, tags in training_data:
       for word in sent:
       if word not in word_to_ix: # word has not been assigned an index yet
       word_to_ix[word] = len(word_to_ix) # Assign each word with a unique index
       print(word_to_ix)
       tag_to_ix = {"DET": 0, "NN": 1, "V": 2} # Assign each tag with a unique index
       # These will usually be more like 32 or 64 dimensional.
       # We will keep them small, so we can see how the weights change as we train.
       EMBEDDING_DIM = 6
       HIDDEN_DIM = 6
       
       
       class LSTMTagger(nn.Module):
def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
super(LSTMTagger, self).__init__()
self.hidden_dim = hidden_dim
self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
# The LSTM takes word embeddings as inputs, and outputs hidden states
# with dimensionality hidden_dim.
self.lstm = nn.LSTM(embedding_dim, hidden_dim)
# The linear layer that maps from hidden state space to tag space
self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
def forward(self, sentence):
embeds = self.word_embeddings(sentence)
lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
tag_scores = F.log_softmax(tag_space, dim=1)
return tag_scores
Train the Model
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


with torch.no_grad():
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)
for epoch in range(300): # again, normally you would NOT do 300 epochs, it is toy data
for sentence, tags in training_data:
# Step 1. Remember that Pytorch accumulates gradients.
# We need to clear them out before each instance
model.zero_grad()
# Step 2. Get our inputs ready for the network, that is, turn them into
# Tensors of word indices.
sentence_in = prepare_sequence(sentence, word_to_ix)
targets = prepare_sequence(tags, tag_to_ix)
# Step 3. Run our forward pass.
tag_scores = model(sentence_in)
# Step 4. Compute the loss, gradients, and update the parameters by
# calling optimizer.step()
loss = loss_function(tag_scores, targets)
loss.backward()
optimizer.step()
# See what the scores are after training
with torch.no_grad():
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

       
       `,

    5: `
    
    import tensorflow as tf
tf.__version__
!pip install imageio
!pip install git+https://github.com/tensorflow/docs
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).asty
pe('float32')
train_images = (train_images -
127.5) / 127.5 # Normalize the images to [-1, 1]
BUFFER_SIZE = 60000
BATCH_SIZE = 256
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(B
UFFER_SIZE).batch(BATCH_SIZE)
ef make_generator_model():
model = tf.keras.Sequential()
model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Reshape((7, 7, 256)))
assert model.output_shape == (None, 7, 7, 256)

model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding=
'same', use_bias=False))
assert model.output_shape == (None, 7, 7, 128)
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='
same', use_bias=False))
assert model.output_shape == (None, 14, 14, 64)
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='s
ame', use_bias=False, activation='tanh'))
assert model.output_shape == (None, 28, 28, 1)
return model
generator = make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')

def make_discriminator_model():
model = tf.keras.Sequential()
model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
input_shape=[28, 28, 1]))
model.add(layers.LeakyReLU())
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
model.add(layers.LeakyReLU())
model.add(layers.Dropout(0.3))
model.add(layers.Flatten())
model.add(layers.Dense(1))
return model
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
real_loss = cross_entropy(tf.ones_like(real_output), real_output)
fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
total_loss = real_loss + fake_loss
return total_loss
def generator_loss(fake_output):
return cross_entropy(tf.ones_like(fake_output), fake_output)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
discriminator_optimizer=discriminator_opt
imizer,
generator=generator,

discriminator=discriminator)

EPOCHS = 5
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])
# Notice the use of "tf.function"
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
noise = tf.random.normal([BATCH_SIZE, noise_dim])
with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
generated_images = generator(noise, training=True)
real_output = discriminator(images, training=True)
fake_output = discriminator(generated_images, training=True)
gen_loss = generator_loss(fake_output)
disc_loss = discriminator_loss(real_output, fake_output)
gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainab
le_variables)
gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminat
or.trainable_variables)
generator_optimizer.apply_gradients(zip(gradients_of_generator, genera
tor.trainable_variables))
discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator
, discriminator.trainable_variables))

def train(dataset, epochs):
for epoch in range(epochs):
start = time.time()
for image_batch in dataset:
train_step(image_batch)
# Produce images for the GIF as you go
display.clear_output(wait=True)
generate_and_save_images(generator,
epoch + 1,
seed)
# Save the model every 15 epochs
if (epoch + 1) % 15 == 0:
checkpoint.save(file_prefix = checkpoint_prefix)
print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-
start))
display.clear_output(wait=True)
generate_and_save_images(generator,
epochs,
seed)

def generate_and_save_images(model, epoch, test_input):
# Notice "training" is set to False.
# This is so all layers run in inference mode (batchnorm).
predictions = model(test_input, training=False)
fig = plt.figure(figsize=(4, 4))
for i in range(predictions.shape[0]):
plt.subplot(4, 4, i+1)
plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
plt.axis('off')
plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
plt.show()
train(train_dataset, EPOCHS)    
    
    `,
    6: `
    
    import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
%matplotlib inline
train =
pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_S
entiment_Analysis/master/train.csv')
train_original=train.copy()
test =
pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_S
entiment_Analysis/master/test.csv')
test_original=test.copy()
combine = train.append(test,ignore_index=True,sort=True)
Combine.tail()
    `,
  };
};
