# Homework 3: Computer Vision, Spring 2024

## Overview

This document provides an overview of the programming assignment for the Computer Vision course. The assignment consists of a programming challenge, focusing on 3d Reconstruction and Reflection based shape recovery.

## Setup and Requirements

- Libraries Used: 
argparse, PIL, numpy, matplotlib, skimage

## Design decisions, Voting Scheme and Algorithm

### Challenge 1a: 
Instead of using a complicated function, since it was mentioned the background is completely black (0,0,0) I used simple thresholding to get the shape. Used regionprops to find the centroid and area using which I obtained the radius. 
Sphere center: [239.50429915, 239.50418544]
Sphere radius: 190.7636698587073

### Challenge 1b:
The approach I took was to try and figure out how an object's surface reflects light from different angles by looking at the object's center and a specific point on its surface in a photo. I imagine a line between these two points to understand how the surface tilts. By analyzing the brightest spots under various lights, I determine the surface's 3D orientation at each point. The intensity of the light at these points helped me calculate the direction and intensity of the lighting, allowing me to find the surface normals and albedo of the object.

This was achieved in th code by centering the object at origin by transposing it and then computed Nz based on distance of point to center. Finally scaling the vectors to the brightest pixel in the image the final surface normal was obtained.

Length of vector joining the center and the point is \\( \sqrt{(P_x - C_x)^2 + (P_y - C_y)^2} \\)
and for Nz we use  
\\[ N_x = P_x - C_x \\]
\\[ N_y = P_y - C_y \\]
\\[ N_z = \sqrt{Radius^2 - N_x^2 - N_y^2} \\]

finally we normalize N.


It's safe to assume the brightest spot indicates the light source's direction due to the Helmholtz Reciprocity principle and the Lambertian reflectance model. The first suggests the path of light remains unchanged even if the light source and observer swap places. Given a Lambertian surface reflects light equally in all directions, the brightest spot directly aligns with the light source, providing a good sense of its direction. This implies that the brightest spot's normal vector points directly back at the light source.

### Challenge 1c:
A simple sum of all image arrays followed by simple thresholding to get the pixels who's values > 0 gives us a reliable mask for the given object as only background points would be consistently 0 in all images. 

### Challenge 1d:
I looped through each pixel, but only focused on the parts where the mask was set to one, essentially pinpointing the object. I applied the formulas from our notes to calculate the albedos and normals. I had to use the pseudo-inverse method so that the code was robust. 
