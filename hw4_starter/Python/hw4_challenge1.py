from PIL import Image, ImageDraw
import numpy as np
from typing import Union, Tuple, List


def computeHomography(src_pts_nx2: np.ndarray, dest_pts_nx2: np.ndarray) -> np.ndarray:
    '''
    Compute the homography matrix.
    Arguments:
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    Returns:
        H_3x3: the homography matrix (3x3 numpy array).
    '''
    n_points = src_pts_nx2.shape[0]
    A = np.zeros((2*n_points, 9))
    for i in range(n_points):
        x, y = src_pts_nx2[i, 0], src_pts_nx2[i, 1]
        xp, yp = dest_pts_nx2[i, 0], dest_pts_nx2[i, 1]
        A[2*i] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        A[2*i + 1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]

    # Compute the matrix A^T A
    ATA = A.T @ A

    # Compute the eigenvalues and eigenvectors of A^T A
    eigenvalues, eigenvectors = np.linalg.eig(ATA)

    # Find the eigenvector corresponding to the smallest eigenvalue
    h = eigenvectors[:, np.argmin(eigenvalues)]

    # Reshape h into the 3x3 homography matrix H
    H_3x3 = h.reshape(3, 3)
    return H_3x3
    raise NotImplementedError


def applyHomography(H_3x3: np.ndarray, src_pts_nx2: np.ndarray) ->  np.ndarray:
    '''
    Apply the homography matrix to the source points.
    Arguments:
        H_3x3: the homography matrix (3x3 numpy array).
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
    Returns:
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    '''
    # Convert source points to homogeneous coordinates
    n_points = src_pts_nx2.shape[0]
    homogeneous_src_pts = np.hstack((src_pts_nx2, np.ones((n_points, 1))))

    # Apply the homography matrix to the source points
    transformed_pts = H_3x3 @ homogeneous_src_pts.T

    # Normalize the points (to make sure the last coordinate is 1)
    normalized_pts = transformed_pts / transformed_pts[-1, :]

    # Convert back to 2D coordinates
    dest_pts_nx2 = normalized_pts[:2, :].T
    return dest_pts_nx2
    raise NotImplementedError


def showCorrespondence(img1: Image.Image, img2: Image.Image, pts1_nx2: np.ndarray, pts2_nx2: np.ndarray) -> Image.Image:
    '''
    Show the correspondences between the two images.
    Arguments:
        img1: the first image.
        img2: the second image.
        pts1_nx2: the coordinates of the points in the first image (nx2 numpy array).
        pts2_nx2: the coordinates of the points in the second image (nx2 numpy array).
    Returns:
        result: image depicting the correspondences.
    '''
    # Create a new image by placing the two images side by side
    result_width = img1.width + img2.width
    result_height = max(img1.height, img2.height)
    result = Image.new('RGB', (result_width, result_height))
    result.paste(img1, (0, 0))
    result.paste(img2, (img1.width, 0))

    # Prepare to draw on the image
    draw = ImageDraw.Draw(result)

    # The points in pts2_nx2 need to be offset by the width of img1
    pts2_nx2_offset = pts2_nx2.copy()
    pts2_nx2_offset[:, 0] += img1.width

    # Draw lines between corresponding points
    for (x1, y1), (x2, y2) in zip(pts1_nx2, pts2_nx2_offset):
        draw.line((x1, y1, x2, y2), fill='red', width=4)
    
    return result
    raise NotImplementedError

# function [mask, result_img] = backwardWarpImg(src_img, resultToSrc_H, dest_canvas_width_height)

def backwardWarpImg(src_img: Image.Image, destToSrc_H: np.ndarray, canvas_shape: Union[Tuple, List]) -> Tuple[Image.Image, Image.Image]:
    '''
    Backward warp the source image to the destination canvas based on the
    homography given by destToSrc_H. 
    Arguments:
        src_img: the source image.
        destToSrc_H: the homography that maps points from the destination
            canvas to the source image.
        canvas_shape: shape of the destination canvas (height, width).
    Returns:
        dest_img: the warped source image.
        dest_mask: a mask indicating sourced pixels. pixels within the
            source image are 1, pixels outside are 0.
    '''
    raise NotImplementedError


def blendImagePair(img1: List[Image.Image], mask1: List[Image.Image], img2: Image.Image, mask2: Image.Image, mode: str) -> Image.Image:
    '''
    Blend the warped images based on the masks.
    Arguments:
        img1: list of source images.
        mask1: list of source masks.
        img2: destination image.
        mask2: destination mask.
        mode: either 'overlay' or 'blend'
    Returns:
        out_img: blended image.
    '''
    raise NotImplementedError

def runRANSAC(src_pt: np.ndarray, dest_pt: np.ndarray, ransac_n: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Run the RANSAC algorithm to find the inliers between the source and
    destination points.
    Arguments:
        src_pt: the coordinates of the source points (nx2 numpy array).
        dest_pt: the coordinates of the destination points (nx2 numpy array).
        ransac_n: the number of iterations to run RANSAC.
        eps: the threshold for considering a point to be an inlier.
    Returns:
        inliers_id: the indices of the inliers (kx1 numpy array).
        H: the homography matrix (3x3 numpy array).
    '''
    raise NotImplementedError

def stitchImg(*args: Image.Image) -> Image.Image:
    '''
    Stitch a list of images.
    Arguments:
        args: a variable number of input images.
    Returns:
        stitched_img: the stitched image.
    '''
    raise NotImplementedError
