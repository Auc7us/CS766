# Homework 4: Computer Vision, Spring 2024

## Overview

This document provides an overview of the programming assignment for the Computer Vision course. The assignment consists of a programming challenge, focusing on image warping and stitching.

## Setup and Requirements

- Libraries Used: 
argparse, PIL, numpy, matplotlib, skimage, tkinter, scipy

## Design decisions

### Challenge 1a: 
Selected the points using the mouse clicks and computed the Homography matrix using these points. I then wrote a fairly straigth forward function to apply the homography on selected inputs. The finally draw the lines between the points.

### Challenge 1b:
I designed this funciton to perform backward warping by mapping each pixel from a destination canvas back to the source image using a given homography matrix. I chose to iterate over the destination canvas, transforming coordinates with the homography to find corresponding pixels in the source image. To indicate which pixels were successfully mapped, I created a binary mask, ensuring clarity on which parts of the transformed image are directly sourced from the original. 

### Challenge 1c:
I implemented the RANSAC algorithm to robustly estimate a homography between source and destination point sets, iteratively selecting random subsets to maximize inliers based on a distance threshold. I had to switch the output obtain from getSIFTmatches to work with the code and draw the points as expected. I emperically chose iterations and eps as 3 and set a random.seed as 7777 for repeatability. 

### Challenge 1d:
For blending the images, I first normalize the masks. In case of overlay i use np.where to replace the part of the image with another. In case of blend i create a mask for each image using distance transform edt on the normalized masks. replace the zeros with 0.000000001 to avoid divide by zero errors and take the weighted average of the images using the masks as the weights.

### Challenge 1e:
When stitching the images I start by loading in the normalized numpy arrays of the images. I then replace the 0,0,0 (true black pixels) with 1/255.0,1/255.0,1/255.0. This helps when creating mask where i can say all the non black pixels are part of image and need to have a mask value of 1 and the other pixels are background. I chose the first image loaded as the image I start with. It can be any part of the final image. Inside the for loop I load in the next picture to stitch. I get the SIFT pairs and run RANSAC to compute the best homography matrix with eps of 0.8 and 1000 iterations. using this H, I compute the coordinates of the warped image and check the required size of the canvas and create the base image and masks accordingly.  Using the above mentioned functions I obtain the blended images after replacing the 1.255.0 pixels with 0.


### Challenge 1f:
Same as part 1e. Used 3 images I shot while in Dubai. The Left and the Right were portrait and the center image is in landscape orientation and the script gave me a really good result.



