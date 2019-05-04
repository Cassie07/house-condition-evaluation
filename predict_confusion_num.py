%matplotlib inline
import os
import numpy as np
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Sequential, Model
from keras import optimizers
import json
import cv2
import os
import numpy as np
from keras.preprocessing import image as Image
from keras.utils import plot_model, to_categorical
import PIL
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten
from keras.models import Model, load_model
from keras.preprocessing import image as Image
from keras.preprocessing import sequence as Sequence
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model, to_categorical
from collections import Counter
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

CUDA_VISIBLE_DEVICES = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
LABELS = ['High','Medium','Low']
lables=np.array(LABELS)


WEIGHTS_PATH = "/projects/checkpoint/weights-improvement-0.5879-0002.hdf5"
#WORDS_PATH = "C:/Users/dingk/Desktop/cse498/assignment3/v19_gru/words.txt"
IMAGE_FILE = "/projects/new"
LABEL_FILE='/projects/test.txt'


IMAGE_SIZE = 224
input_shape=(224,224,3)

class Generate_label(object):

    def __init__(self):#, pra_voc_size):
        #self.classes = classes
        self.model = self.create_model()
        self.model.load_weights(WEIGHTS_PATH)
        self.X_test, self.y_test,self.y_not_onehot = self.get_name_label(LABEL_FILE,IMAGE_FILE)
        #self.tra

    def create_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False,input_shape=input_shape)
        x = base_model.output
        base_model=Model(inputs=base_model.input,outputs=x)

        x = Flatten(name='flatten')(x)
        predictions = Dense(3, activation='softmax', name='predictions')(x) #(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers[0:141]:
            layer.trainable = False

        sgd=optimizers.SGD(lr=0.001,decay=0.0005,momentum=0.7)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def get_name_label(self,DATA_ROOT,IMAGE_ROOT):
        '''
        Load testing data from files.

        Returns:
            X_test: a list of all paths of testing images
            y_test: corresponding testing labels
        '''
        # load all data and split them randomly
        with open(DATA_ROOT, 'r') as reader:
            content = [x.strip().split('\t') for x in reader.readlines()]
            X_test = [os.path.join(IMAGE_ROOT, x[0].split('#')[0]) for x in content]
            y_test=[x[0].split('#')[1] for x in content]
            y_test_not_onehot=[int(i) for i in y_test]
            y_test= to_categorical(y_test)
            #print(X_test)
            #print(y_test)
            #print(train_image_name_list)
        return X_test,y_test,y_test_not_onehot


    def decode(datum):
        return np.argmax(datum)

    def generate(self, image_path):
        # image
        X_test=self.X_test
        y_test=self.y_test
        y_not_onehot=self.y_not_onehot
        input_image_list = []
        input_label_list = []
        #target_label_list = []
        for index, (image_name,image_label) in enumerate(zip(X_test,y_test),start=1):
            # image
            image = load_img(image_name, target_size=(224, 224))
            # convert the image pixels to a numpy array
            image = img_to_array(image)

            # reshape data for the model
            input_image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
            input_image_list.append(input_image)
            input_label_list.append(image_label)
            #target_label_list.append(target_label_onehot)
        X_test = np.array(input_image_list)
        y_test = np.array(input_label_list)
        predict=self.model.predict(X_test)
        pred_index_total=[]
        for i in predict:
            pred_index = []
            pred_list=i.tolist()
            index_max=pred_list.index(max(pred_list))
            pred_index.append(index_max)
            pred_index_total.append(np.array(pred_index))
        #print(pred_index_total)
        one_hot_predictions=to_categorical(np.array(pred_index_total))
        #print("one_hot_predictions%%%%%%%%%",one_hot_predictions)
        prediction=one_hot_predictions.argmax(1)

        #print(prediction)
        #confusion_matrix = metrics.confusion_matrix(y_not_onehot, prediction)
        #print("%%%%%%%%%%%%%%%",confusion_matrix)

        # Plot Results:
        width = 12
        height = 12
        #normalised_confusion_matrix = np.array(confusion_matrix)/np.sum(confusion_matrix)*100
        #print(normalised_confusion_matrix)
        #plt.figure(figsize=(width, height))
        #plt.imshow(
        #    normalised_confusion_matrix,
            #interpolation='nearest',
        #    cmap=plt.cm.rainbow
       # )
        #plt.title("Confusion matrix \n(normalised to % of total test data)")
        #plt.colorbar()
        #tick_marks = np.arange(3)
        #plt.xticks(tick_marks,lables,rotation=90)
        #plt.yticks(tick_marks,lables)
        #plt.tight_layout()
        #plt.ylabel('True label')
        #plt.xlabel('Predicted label')
        #plt.show()
        return prediction,y_not_onehot

label = Generate_label()
prediction,y_not_onehot=label.generate(IMAGE_FILE)
print(prediction)
y_pred=[]
#for i in range(prediction.shape[0]):
#    datum = prediction[i]
#    print(prediction[i])
#    decoded_datum = label.decode(prediction[i])
#    y_pred.append(deconded_datum)
#y_pred=np.array(y_pred)
X_test,y_test,y_test_not_onehot=label.get_name_label(LABEL_FILE,IMAGE_FILE)
print(np.array(y_test_not_onehot))

def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

np.set_printoptions(precision=2)
cm1=plot_confusion_matrix(np.array(y_test_not_onehot), np.array(prediction), classes=lables,title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
cm2=plot_confusion_matrix(np.array(y_test_not_onehot), np.array(prediction), classes=lables, normalize=True,
                              title='Normalized confusion matrix')

plt.show()
