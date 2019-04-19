from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.utils.np_utils import to_categorical
from keras.models import Model, load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

# load image and image names
# all images will be loaded as a 3 dimension img_to_array and added into a list
def load_image_name(path):
    images=[]
    img_name=[]
    files=os.listdir(path)
    for name in files:
        if name=='.DS_Store'or name=='_DS_Store':
            continue
        img_name.append(name)
        #print(name)
        path2=path+'/'+name

        image = load_img(path2, target_size=(224, 224))

        # convert the image pixels to a numpy array
        image = img_to_array(image)

        # reshape data for the model
        image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))

        #img = cv2.imread(path2)
        images.append(image)
    #print(image)
    return img_name,images

# running from here
# images directory
path = '/projects/new'
names,images=load_image_name(path)

# transfer list into an matrix
# shape of the matrix: (2643,224,224,3)
X=np.array(images)

# label
y = np.random.randint(3, size=2643) # 10 labels

# split data into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print(X_train.shape)

# one hot encode outputs

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


# tensorboard
# view the result chart
# view the architecture of NN
# analyse the performance of the model
CHECK_ROOT = 'checkpoint/'
if not os.path.exists(CHECK_ROOT):
    os.makedirs(CHECK_ROOT)
# callback: draw curve on TensorBoard
tensorboard = TensorBoard(log_dir='log', histogram_freq=0, write_graph=True, write_images=True)
# callback: save the weight with the highest validation accuracy
filepath=os.path.join(CHECK_ROOT, 'weights-improvement-{val_acc:.4f}-{epoch:04d}.hdf5')
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')

# create model
input_shape = (224, 224, 3)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
#base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
nnum_classes=3
#num_pixel=640*640
# design model
#model = Sequential()
#model.add(Conv2D(32, (5, 5), input_shape=(224,224,3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dense(num_classes, activation='softmax'))

# print out summary of the model
#print(model.summary())

x = base_model.output
x = Flatten(name='flatten')(x)
predictions = Dense(3, activation='softmax', name='predictions')(x) #(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[0:141]:
    layer.trainable = False

# print out summary of the model
print(model.summary())

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2,callbacks=[tensorboard, checkpoint])


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

print("Baseline Error: %.2f%%" % (100-scores[1]*100))
