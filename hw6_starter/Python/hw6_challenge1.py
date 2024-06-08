from PIL import Image
import numpy as np
from typing import Union, Tuple, List
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import scipy.ndimage

def generateIndexMap(gray_list: List[np.ndarray], w_size: int) -> np.ndarray:
    # Generate an index map for the refocusing application
    # Input:
    #   gray_list - List of K gray-scale images
    #   w_size - half window size used for focus measure computation
    # Output:
    #   index_map - mxn index map
    #               index_map(i, j) is the index of the image that is in focus
    #               at pixel (i, j)
    
    K = len(gray_list) 
    w, h = gray_list[0].size
    
    focus_stack = np.zeros((h, w, K))
    
    for idx, gray in enumerate(gray_list):
        focus_stack[:, :, idx] = focus_measure(gray, 2*w_size+1)

    index_smooth = apply_moving_average(focus_stack, 10)    
    
    index_map = np.argmax(index_smooth, axis=2)
    
    index_map = scipy.ndimage.median_filter(index_map, size = 21)
    
    return index_map
    raise NotImplementedError

def loadFocalStack(focal_stack_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Load the focal stack
    # Input:
    #   focal_stack_dir - directory of the focal stack
    # Output:
    #   rgb_list - List of RGB images for varying focal lengths
    #   gray_list - List of gray-scale images for varying focal lengths
    rgb_list = []
    gray_list = []
    filenames = sorted(os.listdir(focal_stack_dir), key=lambda x: int(x.replace('frame','').replace('.jpg','')))
    for filename in filenames:
        if filename.endswith(".jpg"):
            file_path = os.path.join(focal_stack_dir, filename)
            img = Image.open(file_path)
            img_g = img.convert('L')
            rgb_list.append(img)
            gray_list.append(img_g)
    return rgb_list, gray_list
    raise NotImplementedError
    
def apply_moving_average(focus_stack: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    smoothed_stack = np.empty_like(focus_stack)
    for idx in range(focus_stack.shape[2]):
        smoothed_stack[:, :, idx] = convolve2d(focus_stack[:, :, idx], kernel, mode='same', boundary='wrap')
    return smoothed_stack

def focus_measure(image: np.ndarray, window_size: int) -> np.ndarray:
    # Create the Laplacian kernel (second derivative approximation)
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    
    # Convolve the image with the Laplacian kernel
    laplacian_response = convolve2d(image, laplacian_kernel, mode='same', boundary='symm')
    
    # Square the response and sum over the window
    squared_response = np.square(laplacian_response)
    focus_m = convolve2d(squared_response, np.ones((window_size, window_size)), mode='same', boundary='symm')
    return focus_m

def refocusApp(rgb_list: List[np.ndarray], depth_map: np.ndarray) -> None:
    # Refocusing application
    # Input:
    #   rgb_list - List of RGB images for varying focal lengths
    #   depth_map - mxn index map
    #               depth_map(i, j) is the index of the image that is in focus
    #               at pixel (i, j)
    

    fig, ax = plt.subplots()
    im = ax.imshow(rgb_list[0])  # Display the first image in the list

    while True:
        point = plt.ginput(1, timeout=-1, show_clicks=False)
        if not point:
            break

        scene_point = tuple(map(int, point[0]))

        if (0 <= scene_point[0] < depth_map.shape[1]) and (0 <= scene_point[1] < depth_map.shape[0]):
            i, j = scene_point
            focal_index = depth_map[j, i]

            refocused_image = rgb_list[int(focal_index)]

            im.set_data(refocused_image)
            plt.draw()
            plt.pause(0.1) 
        else:
            print("Selected point is outside the image.")

    plt.close()
    
    
    
    raise NotImplementedError

