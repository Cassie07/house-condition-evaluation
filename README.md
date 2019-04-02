house-condition-evaluation

source code(Especially image segmentation and network architecture part)
reference: https://coursesite.lehigh.edu/pluginfile.php/2315021/mod_resource/content/1/Visual%20Estimation%20of%20Building%20Condition%20with%20Patch-level%20ConvNets.pdf

dataset: image downloads by google street view API
         address provided by Allentown government
         
Method: image segmentation(SIFT, cluster algorithms)    
        prediction(ResNet50)

Project Member: Kexin, Nathan, Yue

sitf_kmeans_2.py: image segmentation which uses sitf to extract features and using kmeans to cluster.
                  What's more, we use a second kmeans for the keypoints in each cluster which get from the former steps.
                  In this step, we just cluster each keypoint by its coordinate in this step.(Only Physical Distance)

