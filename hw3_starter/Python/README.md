# Homework 3: Computer Vision, Spring 2024

## Overview

This document provides an overview of the programming assignment for the Computer Vision course. The assignment consists of a programming walkthrough and a programming challenge, focusing on image processing and edge detection techniques.

## Setup and Requirements

- Libraries Used: 
argparse, PIL, numpy, matplotlib, skimage, math, scipy

## Design decisions, Voting Scheme and Algorithm

### Walkthrough 1: 
Fixed erros in plotting subplots and found the thresholds empirically.

### Challenge 1a: 
Opted for Canny over Sobel as I prefer single pixel wide edges and Canny also delivered better results.
Empirically tuned the values of thrasholds 

### Challenge 1b:
Jed Pully and I worked on this together. After noticing that basic voting tends to miss a few lines, to better the contrast of the accumulators we decided to move forward with soft voting using a 3x3 matrix. Where the votes were again chosen through trial and error. We initially ran into trouble when we limited our theta range to 180 where we'd miss a few lines that went over this value. Changing the range to 360 fixed the issue. I also used rho = c*np.sin(t*np.pi/180) + r*np.cos(t*np.pi/180) since the theta I chose is the complementary angle of the theta used in the slides. Changing the bin sizes to be greater than 1 gave us poor results so we stuck to bin width of 1 for both rho and theta.

### Challenge 1c:
Basic thresholding always delivered bad results. After realising that sort of adaptive equalisation was required before we could use a final treshold, the image was first passed through a smaller threshold acting as a high pass filter. This was split into multiple smaller rectangular parts where the pixels in each part were then scaled to the range 0,255. This was done to make the peaks sharper as a final high treshold would give a really good result with only few lines per edge. The number of parts was chosen after examining the high pass filtered accumulator. To avoid saving and importing multiple images in 1c most of this was integrated into 1b and only the final threshold along with the line drawing function was implemented in 1c. I drew thick lines of 3px width for better visualization.

### Challenge 1d:
Since we were asked not to use cv2.HoughLinesP directly, I looked into its implementation. After going through the algorithm, reimplimenting it from scratch would mean ignoring the results in 1c. My basic take away was that after sparsly sampling the points from edge image, it tries to find line segments and sorts the points that belong to the same line before picking the ends as the end points of line segment. Given the nature of our input images that have several line segments on the same line, it would required an additional step of filtering points and keeping track of line segments. Since we already have most of the information in the edge image and have found the lines in 1c, I used the lines as a mask after dilating them. A bitwise AND with the edge mask would give us all the line segments purging the infinite line. This however resulted in few artifacts that required pruning. Taking inspiration from HoughLinesP, I sampled random points from the resulting image and examined the 15x15 region surrounding it. If atleast one pixel on two edges of this region is white, it meant it was part of a long edge. If less than two edges have only pixel as white then it means we have encounted a small artifact that needs to be pruned. I over-write all such regions with 0s and display outputs, which I found to be satisfactory.

