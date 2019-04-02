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
img = cv2.imread("/projects/house3.jpg", cv2.IMREAD_GRAYSCALE)

#w, h= original_shape = tuple(img.shape)
#print(w)
#print(h)

# SIFT
sift = cv2.xfeatures2d.SIFT_create()
# Get the keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(img, None)
# test: get the coordinates of each keypoints
coor=keypoints[0].pt
#print(coor[0])
#print(descriptors)
# show the keypoints on the image
img = cv2.drawKeypoints(img, keypoints, None)
cv2.imshow("Image_1", img)
print(descriptors)

# Kmeans for keypoints
kmeans = KMeans(n_clusters=20, random_state=10).fit(descriptors)
# Get the centroid of each cluster
centers = np.array(kmeans.cluster_centers_)
print(tuple(centers.shape))

# plot centroids
plt.plot()
plt.title('k means centroids')
plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')
plt.show()
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
for i in dic.values():
    X=[]
    for j in i:
        X.append([j[0],j[1]])
    X = np.array(X)
    # kmeans: clustering by using coordinates(Physical distance)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    centers = np.array(kmeans.cluster_centers_)
    print(centers)
    # Select the size of cropping box
    for j in centers:
        x_min=j[0]-100
        x_max=j[0]+100
        y_min=j[1]-100
        y_max=j[1]+100
        img = "/projects/house3.jpg"
        name='img'+str(index)+'.jpg'
        path="/projects/seg"
        crop(img, (x_min, y_min, x_max, y_max), name)
        index+=1
    plt.plot()
    plt.title('k means centroids')
    plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')
    plt.show()
    # Get labels for all points
    print("Predicting color indices on the full image (k-means)")
    t0 = time()
    
    # Another segment method
    #cluster_labels = kmeans.predict(X)
    #keypoints_cluster,dic_cluster=get_dic(cluster_labels)
    #for j in dic_cluster.values():
    #    print(j)
    #    x_min=10000
    #    x_max=0
    #    y_min=10000
    #    y_max=0
    #    for m in j:
    #        if m[0]<x_min:
    #            x_min=m[0]
    #        elif m[0]>x_max:
    #            x_max=m[0]
    #        if m[1]<y_min:
    #            y_min=m[1]
    #        elif m[1]>y_max:
    #            y_max=m[1]
    #    print(x_min)
    #    print(y_max)
    #    print(x_max)
    #    print(y_min)
    #    img = "/projects/house3.jpg"
    #    name='img'+str(index)+'.jpg'
    #    path="/projects/seg"
    #    index+=1

#print(dic)
