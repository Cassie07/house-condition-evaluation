% matplotlib inline
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
house_name='/projects/house_test.txt'
patch_name='/projects/patch_test.txt'
LABELS=['High','Medium','Low']
labels=np.array(LABELS)

def get_name_label(DATA_ROOT):
    '''
    Load testing data from files.

    Returns:
        X_test: a list of all paths of testing images
        y_test: corresponding testing labels
    '''
    # load all data and split them randomly
    with open(DATA_ROOT, 'r') as reader:
        content = [x.strip().split('\t') for x in reader.readlines()]
        X_test = [x[0].split('#')[0] for x in content]
        y_test=[x[0].split('#')[1] for x in content]
        #y_test_not_onehot=[int(i) for i in y_test]
        #y_test= to_categorical(y_test)
        #print(X_test)
        #print(y_test)
        #print(train_image_name_list)
    return X_test,y_test

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    print(classes)
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

X_house_test,y_house_test= get_name_label(house_name)
X_patch_test,y_patch_test= get_name_label(patch_name)

# dictionary of house and its labels
house_ground_truth={}
house_name=[]
for (X_test,y_test) in zip(X_house_test,y_house_test):
    house_ground_truth[X_test]=y_test
house_names=[i for i in house_ground_truth.keys()]
#print(house_names)


# dictionary of patch and its labels
patch_ground_truth={}
patch_names=[]
for (X_test,y_test) in zip(X_patch_test,y_patch_test):
    patch_ground_truth[X_test]=y_test
    patch_names.append(X_test)
#print(patch_names)


# add y_pred
dict_patch={}
with open('/Users/kexinding/Downloads/y_pred.txt','r') as reader:
    content =[x.strip().split('\t') for x in reader.readlines()]
    y_pred=[int(i[0]) for i in content]
    #y_pred=np.array(y_pred)
#y= np.random.randint(3, size=8825)
#y_pred=y.tolist()
for (i,j) in zip(patch_names,y_pred):
    dict_patch[i]=int(j)
print(dict_patch)

# {'house_name':['labels of patches']}
dict_house={}
for i in house_names:
    dict_house[i]=[]
for key,value in dict_patch.items():
    name=key.split('_')[0]+'.jpg'
    if name in house_names:
        dict_house[name].append(int(value))

        
label=[0,1,2]
for key,value in dict_house.items():
    print('value')
    print(value)
    max_list=[]
    for i in label:
        n=value.count(i)
        print(n)
        max_list.append(n)
    print(max_list)
    max_num=max(max_list)
    dict_house[key]=max_list.index(max_num)
    
# compare
y_test=[int(i) for i in house_ground_truth.values()]
y_pred=[int(i) for i in dict_house.values()]    
print(y_test)
print(y_pred)
y_test=np.array(y_test)
y_pred=np.array(y_pred)



np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=labels,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=labels, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
