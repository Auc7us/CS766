from PIL import Image, ImageDraw
import numpy as np

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
                    voter = np.array([[0.06, 0.06, 0.06,],[0.06, 0.9, 0.06],[0.06, 0.06, 0.06]])
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
    
    height = np.shape(orig_img)[0] 
    width = np.shape(orig_img)[1] 

    for r, t in zip(hough_peaks_r,hough_peaks_c):
        if r == 0 :
            rho = ((r)*RHO_BIN_WID)
        else:
            rho = ((r)*RHO_BIN_WID)
        if t == 0:
            t = 1
        theta = (np.pi/180)*((t)*THETA_BIN_WID)
        xp0, yp0, xp1, yp1 =  rho/np.sin(theta), 0, 0, rho/np.cos(theta)
        if yp1 < 0: #r' is negative
            xp1, yp1 =  (rho-height*np.cos(theta))/np.sin(theta), height
        if xp0 < 0:
            xp0, yp0 =  (rho-height*np.cos(theta))/np.sin(theta), height
        # print(rho,theta*180/np.pi)
        draw.line((xp0, yp0, xp1, yp1), fill=128)
        # draw.line((-100, yp0, 600, 1200), fill=128)
    
    line_img.show()
    return  line_img
    raise NotImplementedError

def lineSegmentFinder(orig_img: np.ndarray, edge_img: np.ndarray, hough_img: np.ndarray, hough_threshold: float):
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
    raise NotImplementedError
