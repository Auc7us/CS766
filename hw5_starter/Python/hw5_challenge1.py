from PIL import Image
import numpy as np
from typing import Union, Tuple, List
from skimage.measure import label, regionprops

def findSphere(img: np.ndarray) -> Tuple[np.ndarray, float]:
    # Find the center and radius of the sphere
    # Input:
    #   img - the image of the sphere
    # Output:
    #   center - 2x1 vector of the center of the sphere
    #   radius - radius of the sphere
    img = img > 0
    labeled_image = label(img)
    props = regionprops(labeled_image)
    prop = props[0]  
    center_y, center_x = prop.centroid
    center = np.array([center_y, center_x])
    radius = np.sqrt(prop.area / np.pi)
    Image.fromarray(img).show()
    return center, radius
    raise NotImplementedError

def computeLightDirections(center: np.ndarray, radius: float, images: List[np.ndarray]) -> np.ndarray:
    # Compute the light source directions
    # Input:
    #   center - 2x1 vector of the center of the sphere
    #   radius - radius of the sphere
    #   images - list of N images
    # Output:
    #   light_dirs_5x3 - 5x3 matrix of light source directions
    
    light_dirs = []
    
    for img in images:
        # Find the brightest spot's coordinates
        bright_y, bright_x = np.unravel_index(np.argmax(img), img.shape)
        
        # Compute the normal vector components in the image plane
        N_x = bright_x - center[0]
        N_y = bright_y - center[1]
        
        # Compute the z-component of the normal vector
        N_z = np.sqrt(radius**2 - N_x**2 - N_y**2)
        
        # Normalize the normal vector
        norm = np.sqrt(N_x**2 + N_y**2 + N_z**2)
        N = np.array([N_x, N_y, N_z]) / norm
        
        # Scale the normal vector by the brightness of the brightest spot
        brightness = img[bright_y, bright_x]
        N_scaled = N * brightness
        
        light_dirs.append(N_scaled)
    
    return np.array(light_dirs)
    
    raise NotImplementedError

def computeMask(images: List[np.ndarray]) -> np.ndarray:
    # Compute the mask of the object
    # Input:
    #   images - list of N images
    # Output:
    #   mask - HxW binary mask
    mask = np.zeros_like(images[0])
    for i in range(len(images)):
        images[i] = images[i] > 0
        mask += images[i]
        
    mask = mask > 0
    return mask
    raise NotImplementedError

def computeNormals(light_dirs: np.ndarray, images: List[np.ndarray], mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Compute the surface normals and albedo of the object
    # Input:
    #   light_dirs - Nx3 matrix of light directions
    #   images - list of N images
    #   mask - binary mask
    # Output:
    #   normals - HxWx3 matrix of surface normals
    #   albedo_img - HxW matrix of albedo values
    
    # Initialize the output images
    H, W = mask.shape
    normals = np.zeros((H, W, 3), dtype=np.float32)
    albedo_img = np.zeros((H, W), dtype=np.float32)
    
    stacked_imgs = np.stack(images, axis=-1)

    for r in range(H):
        for c in range(W):
            if mask[r, c]:
                intensity = stacked_imgs[r, c, :].astype(np.float64)
                norm_vec = np.linalg.pinv(light_dirs) @ intensity
                albedo_val = np.linalg.norm(norm_vec)
                norm_unit = norm_vec / albedo_val
                normals[r, c] = norm_unit
                albedo_img[r, c] = albedo_val

    return normals, albedo_img    
    raise NotImplementedError

