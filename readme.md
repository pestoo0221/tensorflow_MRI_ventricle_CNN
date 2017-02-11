Using Encode-Decoder Deep Convolutional Neural Networks to Segment Ventricle from MR Image (TENSORFLOW) 

**Overview**
The application of CNN in extracing ventricles from structural MR images. 

**Data set** 

* MRI DICOM 3D image from 118 subjects. 
* This is a 2D CNN implementation. For each subject's 3D image, I extracted ~34 2D images on the sagittal view where ventricle is present. In total I have 4032 images.
* All the images were preprocessed by Freesurfer for image intensity normalization and to label the ventricles. (so my label information is based on existing algorithm, not manual label. But this is just a demonstration that how well this CNN works) 
* For each image, the original size is 256 x 256. It is too big for my computer to run the deep learning. So I rescaled the images to 96 x 96. Image intesity is scaled to 0~1 for a better performance. Label information is either 0 (not ventricle) or 1 (ventricle). 

**Model**

1) Unet  [1] 
2) ED-CNN [2] 

We use negative dice coefficient as the loss to train the model. So if the loss is -1, the match will be perfect match. The dice coefficient is calculated as the intersection of true label image and the predicted image, divided by the union of the two. 

1) Unet : This implements with dice coefficient of close to 0.45 for testing images. With 3200 training images, I ran 100 image per batch, and 50 Epochs, it took 5 minutes per epoch with 8 CPU. 
2) ED-CNN : This implements with dice coefficient of around0.45 for testing images. With 4000 training images, I ran 100 image per batch, and 50 Epochs, it took 5 minutes per epoch with 8 CPU. 

**Note**

* Image files are not included due to privacy. Please extract useful code for you own adaption.
* test.py doesn't have any useful code. was some file for me to look for the current directory. There are other ways to avoid using this file.

**Summary reported on tensorboard**
please change the log location ('/home/jidan/test/train') to your own.