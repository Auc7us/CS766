# Homework 2: Computer Vision, Spring 2024

## Overview

This document provides an overview of the programming assignment for the Computer Vision course, due on February 16, 2024. The assignment consists of a programming walkthrough and a programming challenge, focusing on image processing and object recognition techniques.

## Setup and Requirements

- Libraries Used: 
argparse
PIL 
numpy
matplotlib
skimage
math



## Thresholds and Combination Criteria

Walkthrough 1: 
Fixed erros and picked kernel sizes of 7 and 25 to denoise and remove rice respectively

Challenge 1a: 
The default threshold 0.5, 0.5, 0.5 appears to work fine. 

Challenge 1b:
Used nested for loops to compute the 2D properties and stored them in an array in the following order  before displayign them and using them to draw the center and the axis
-object label
-x coordinate of center
-y coordinate of center
-min moment
-axis
-roundness
-area

Challenge 1c:
Used the fucntion created in the above step to create test image's object database.
The ratio of Roundness of the two obbjects and Min Moment of the two objects were used to compare the properties.
an error of 10% was considered acceptable for the ratio of roundness while an error 20% was used for Min Moment (a higher tolerance coz that the moment would change with the change in scale of the image)
Finally displayed and saved the images with the recognised object's center and axis.
