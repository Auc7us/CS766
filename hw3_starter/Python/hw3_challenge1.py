from PIL import Image, ImageDraw
import numpy as np
from skimage.morphology import dilation
from skimage import filters

THETA_BIN_WID = 1
RHO_BIN_WID = 1

def generateHoughAccumulator(edge_image: np.ndarray, theta_num_bins: int, rho_num_bins: int) -> np.ndarray:
    '''
    Generate the Hough accumulator array.
    Arguments:
        edge_image: the edge image.
        theta_num_bins: the number of bins in the theta dimension.
        rho_num_bins: the number of bins in the rho dimension.
    Returns:
        hough_accumulator: the Hough accumulator array.
    '''
    hough_accumulator = np.zeros((rho_num_bins, theta_num_bins))
    
    nRow = np.shape(edge_image)[0] # x
    nCol = np.shape(edge_image)[1] # y
    
    hRow = np.shape(hough_accumulator)[0]
    hCol = np.shape(hough_accumulator)[1]
    
    
    for r in range(nRow):
        for c in range(nCol):
            if edge_image[r][c] == 255:
                # rho = (r**2 + c**2)**(0.5)
                # theta = np.arccos(r/rho) * (180/np.pi)
                for t in range(0, 360, THETA_BIN_WID):
                    # rho = r*np.sin(t) - c*np.cos(t)
                    rho = c*np.sin(t*np.pi/180) + r*np.cos(t*np.pi/180)
                    rho_index = int(rho / RHO_BIN_WID)
                    theta_index = int(t / THETA_BIN_WID)
                    # voter = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],[1.0, 2.0, 2.0, 2.0, 1.0],[1.0, 2.0, 15, 2.0, 1.0],[1.0, 2.0, 2.0, 2.0, 1.0],[1.0, 1.0, 1.0, 1.0, 1.0]])
                    voter = np.array([[0.06, 0.06, 0.06,],[0.06, 0.75, 0.06],[0.06, 0.06, 0.06]])
                    # voter = np.ones((5,5))
                    px, py = rho_index-int(voter.shape[0]/2), theta_index-int(voter.shape[1]/2)
                    
                    # Check bounds
                    if px >= 0 and py >= 0 and px+voter.shape[0] <= hough_accumulator.shape[0] and py+voter.shape[1] <= hough_accumulator.shape[1]:
                        hough_accumulator[px:px+voter.shape[0], py:py+voter.shape[1]] += voter
                    elif 0 <= rho_index < hough_accumulator.shape[0] and 0 <= theta_index < hough_accumulator.shape[1]:
                        # If the voter matrix can't be fully applied, fall back to incrementing the center bin
                        hough_accumulator[rho_index, theta_index] += 1

                    # hough_accumulator[int(rho/RHO_BIN_WID), int(t/THETA_BIN_WID)] += 1
    maxAcc = np.max(hough_accumulator)
    print(maxAcc)
    hough_accumulator = hough_accumulator * (255/maxAcc)
    return hough_accumulator
    raise NotImplementedError


def lineFinder(orig_img: np.ndarray, hough_img: np.ndarray, hough_threshold: float):
    '''
    Find the lines in the image.
    Arguments:
        orig_img: the original image.
        hough_img: the Hough image.
        hough_threshold: the threshold for the Hough accumulator array.
    Returns: 
        line_img: PIL image with lines drawn.

    '''
    hough_im2 = hough_img > hough_threshold
    
    hough_peaks_r,hough_peaks_c = np.where(hough_im2 > 0)
    print(zip(hough_peaks_r,hough_peaks_c))
    line_img = Image.fromarray(orig_img.astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(line_img)
    blank_space = Image.fromarray(np.zeros_like(line_img).astype(np.uint8))
    getLines = ImageDraw.Draw(blank_space)
    
    height = np.shape(orig_img)[0] 
    width = np.shape(orig_img)[1] 

    for r, t in zip(hough_peaks_r,hough_peaks_c):
        rho = ((r)*RHO_BIN_WID)
        if t == 0:
            xp0, yp0, xp1, yp1 =  0, rho, width, rho
        else:
            theta = (np.pi/180)*((t)*THETA_BIN_WID)
            xp0, yp0, xp1, yp1 =  rho/np.sin(theta), 0, 0, rho/np.cos(theta)
            if yp1 < 0: #r' is negative
                xp1, yp1 =  (rho-height*np.cos(theta))/np.sin(theta), height
            if xp0 < 0:
                xp0, yp0 =  (rho-height*np.cos(theta))/np.sin(theta), height
        # print(rho,theta*180/np.pi)
        draw.line((xp0, yp0, xp1, yp1), fill=128,width=3)
        getLines.line((xp0, yp0, xp1, yp1), fill="white")
        # draw.line((-100, yp0, 600, 1200), fill=128)
    
    line_img.show()
    return  line_img, blank_space
    raise NotImplementedError

def lineSegmentFinder(orig_img: np.ndarray, edge_img: np.ndarray, just_lines: np.ndarray):
    '''
    Find the line segments in the image.
    Arguments:         
        orig_img: the original image.
        edge_img: the edge image.
        hough_img: the Hough image.
        hough_threshold: the threshold for the Hough accumulator array.
    Returns:
        line_segement_img: PIL image with line segments drawn.
    '''
    original_rgb = np.stack([orig_img]*3, axis=-1)
    red_overlay = np.array([255, 0, 0], dtype=np.uint8)
    
    k = 7
    selem1 = np.ones((k, k))
    processed_lines = dilation(just_lines, selem1)
    line_mask = processed_lines & edge_img
    binary_mask = line_mask.astype(bool)
    binary_copy = np.copy(binary_mask)

    def process_random_points(image, image_copy, block_size, num_samples):

        height, width = image.shape
        
        # Identify all white points
        white_points = np.argwhere(image == 1)
        print(len(white_points))
        
        # Randomly sample white points
        if len(white_points) > num_samples:
            sampled_points = white_points[np.random.choice(len(white_points), size=num_samples, replace=False)]
        else:
            sampled_points = white_points
        
        # Process each sampled point
        for y, x in sampled_points:
            start_x = max(x - block_size // 2, 0)
            start_y = max(y - block_size // 2, 0)
            end_x = min(x + block_size // 2 + 1, width)
            end_y = min(y + block_size // 2 + 1, height)

            kernel = image[start_y:end_y, start_x:end_x]
            
            if kernel.shape[0] < block_size or kernel.shape[1] < block_size:
                continue
            
            edge_count = 0
            if np.any(kernel[0, :]):
                edge_count += 1
            if np.any(kernel[:, 0]):
                edge_count += 1
            if np.any(kernel[-1, :]):
                edge_count += 1
            if np.any(kernel[:, -1]):
                edge_count += 1
            
            if edge_count < 2:
                image_copy[start_y:end_y, start_x:end_x] = 0
        
        return image_copy
   
    block_size = 15
    num_samples = 1500
    processed_mask = process_random_points(binary_mask, binary_copy, block_size, num_samples)
    segemnted_lines = np.where(processed_mask[..., None], red_overlay, original_rgb)
    
    final_image = Image.fromarray(segemnted_lines)
    final_image.show()


    return final_image
    raise NotImplementedError
