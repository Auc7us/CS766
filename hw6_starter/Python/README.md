#Homework 6: Computer Vision, Spring 2024
##Overview

This document outlines the programming assignment for the Computer Vision course, focusing on digital refocusing using a focal stack of images. The assignment entails a programming challenge that leverages advanced image processing techniques to manipulate the focus of images post-capture.
##Setup and Requirements
-Libraries Used

    PIL: For image loading, conversion between RGB and grayscale, and general image manipulation.
    numpy: For efficient numerical computations and handling arrays of image data.
    matplotlib: For displaying images and interacting with them through clicks.
    scipy: Specifically for the convolve2d function, used in calculating the focus measure of images and applying a moving average filter.

##Design Decisions

###Challenge 1: Generating an Index Map for Refocusing

I implemented a series of functions to create an index map indicating the most in-focus image at each pixel location across a stack of images. This involved computing a focus measure using the Laplacian filter to identify edges and areas of high contrast, which are indicative of focus. A moving average was applied to smooth the focus measures and reduce noise. The index map was generated by selecting the image with the highest focus measure at each pixel.

###Challenge 2: Loading and Preparing the Focal Stack

For this challenge, I developed a function to load a focal stack from a directory, converting all images to both RGB and grayscale. The images were sorted by filename, which corresponds to their order in the focal stack, facilitating the refocusing process based on user interaction.

###Challenge 3: Refocusing Application

I crafted an interactive application allowing users to refocus images based on their depth map. Utilizing matplotlib for image display and interaction, the application updates the focus in real-time based on user clicks. This demonstrates the practical application of computational photography techniques in manipulating focus after image capture.

##Additional Notes

-Homography and Warping: While not directly implemented in this assignment, the concepts of homography and image warping are relevant to understanding how focus can be manipulated in a computational photography context. These techniques underpin the ability to change the perspective and focus of an image post-capture.
-Empirical Choices: The window size for focus measure computation and the kernel size for the moving average were chosen based on empirical testing to balance the accuracy of focus detection and computational efficiency.
-User Interaction: The refocusing application is designed to be intuitive, allowing users to click on the area of interest to adjust focus, demonstrating the practical implications of digital refocusing technology.