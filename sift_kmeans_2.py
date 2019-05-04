# matplotlib inline can help to plot image in Jupyter Notebook
% matplotlib inline
import cv2
import numpy as np
from PIL import Image
import skimage.data as skid
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
import matplotlib.image as img

# Load image
img = cv2.imread("/projects/house2.jpg", cv2.IMREAD_GRAYSCALE)
w, h= original_shape = tuple(img.shape)

# SIFT(Extract Feature)
sift = cv2.xfeatures2d.SIFT_create()
# Get the keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(img, None)
# test: get the coordinates of each keypoints
#coor=keypoints[0].pt

# show the keypoints on the image
img = cv2.drawKeypoints(img, keypoints, None)
cv2.imshow("Image_1", img)
print(descriptors)

# Kmeans for keypoints
kmeans = KMeans(n_clusters=80, random_state=10).fit(descriptors)
# Get the centroid of each cluster
centers = np.array(kmeans.cluster_centers_)
print(tuple(centers.shape))

# plot centroids
#plt.plot()
#plt.title('k means centroids')
#plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')
#plt.show()
#neigh = NearestNeighbors(n_neighbors=3).fit(descriptors)

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(descriptors)
#labels = db.labels_
print("done in %0.3fs." % (time() - t0))

# function for cropping images
def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    #cropped_image.show()

# function to get dictionaries
# input: labels(a list of labels for each keypoints)
# output: keypoints_new(the clustered result for each keypoints. {'0':[keypoint1,keypoint2,...],'1':[keypoint3,keypoint4,...],...})
#         dic(the coordinates of each keypoints. {'0':[(234,1),(37,345),...],'1':[(...),(...),...],...})
keypoints_new={}
dic={}
index=0

def get_dic(labels):
    keypoints_new={}
    dic={}
    index=0
    for i in range (len(labels)):
        label=labels[i]
        if label in keypoints_new.keys():
            dic[label].append(keypoints[i].pt)
            keypoints_new[label].append(keypoints[i])
        else:
            dic[label]=[]
            keypoints_new[label]=[]
            dic[label].append(keypoints[i].pt)
            keypoints_new[label].append(keypoints[i])
    return keypoints_new,dic

keypoints_new,dic=get_dic(labels)
#print(dic)


# Segment
index=0
for key,i in dic.items():
    print('cluster')
    print(key)
    X=[]
    for j in i:
        X.append([j[0],j[1]])
    X = np.array(X)
    # kmeans: clustering by using coordinates(Physical distance)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    centers = np.array(kmeans.cluster_centers_)
    y=kmeans.predict(X)
    print(centers)
    # Select the size of cropping box
    for j in centers:
        if j[0]>=100:
            x_min=j[0]-100
        else:
            x_min=0
        if j[0]+100<=w:
            x_max=j[0]+100
        else:
            x_max=w
        if j[1]>=100:
            y_min=j[1]-100
        else:
            y_min=0
        if j[1]+100<=w:
            y_max=j[1]+100
        else:
            y_max=h
        img = "/projects/house2.jpg"
        name='/'+'projects'+'/'+'res'+'/'+str(index)+'.jpg'
        path="/projects/seg"
        crop(img, (x_min, y_min, x_max, y_max), name)
        index+=1
    plt.plot()
    #plt.title('k means centroids')
    #plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')
    #plt.show()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="plasma")
    plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1],marker='^', c=[0, 1, 2, 3],s=100,linewidth=2,cmap="plasma")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
    # Get labels for all points
    #print("Predicting color indices on the full image (k-means)")
    #t0 = time()
    #img = cv2.imread("/projects/house3.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.drawKeypoints(img, keypoints[1], None)
    #cv2.imshow("Image_1", img)
    #print(descriptors)
    
# draw keypoints on image
img = cv2.imread("/projects/house2.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.drawKeypoints(img, keypoints_new[1], None)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
