from PIL import Image, ImageDraw
import numpy as np
from typing import Union, Tuple, List
from scipy.ndimage import distance_transform_edt
from helpers import genSIFTMatches

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
    ATA = A.T @ A
    eigenvalues, eigenvectors = np.linalg.eig(ATA)
    h = eigenvectors[:, np.argmin(eigenvalues)]
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
    n_points = src_pts_nx2.shape[0]
    homogeneous_src_pts = np.hstack((src_pts_nx2, np.ones((n_points, 1))))
    transformed_pts = H_3x3 @ homogeneous_src_pts.T
    normalized_pts = transformed_pts / transformed_pts[-1, :]
    dest_pts_nx2 = normalized_pts[:2, :].T
    return dest_pts_nx2
    raise NotImplementedError


def showCorrespondence(img1: Image.Image, img2: Image.Image, pts1_nx2: np.ndarray, pts2_nx2: np.ndarray) -> np.ndarray:
    '''
    Show the correspondences between the two images.
    Arguments:
        img1: the first image.
        img2: the second image.
        pts1_nx2: the coordinates of the points in the first image (nx2 numpy array).
        pts2_nx2: the coordinates of the points in the second image (nx2 numpy array).
    Returns:
        res: image depicting the correspondences.
    '''
    
    res_width = img1.width + img2.width
    res_height = max(img1.height, img2.height)
    res = Image.new('RGB', (res_width, res_height))
    res.paste(img1, (0, 0))
    res.paste(img2, (img1.width, 0))

    draw = ImageDraw.Draw(res)

    pts2_nx2_offset = pts2_nx2.copy()
    pts2_nx2_offset[:, 0] += img1.width

    for (x1, y1), (x2, y2) in zip(pts1_nx2, pts2_nx2_offset):
        draw.line((x1, y1, x2, y2), fill='red', width=4)
    
    return res
    raise NotImplementedError

# function [mask, res_img] = backwardWarpImg(src_img, resToSrc_H, dest_canvas_width_height)

def backwardWarpImg(src_img: np.ndarray, destToSrc_H: np.ndarray, canvas_shape: Union[Tuple, List]) -> Tuple[np.ndarray, np.ndarray]:
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
    dest_height, dest_width = canvas_shape
    dest_img = np.zeros((dest_height, dest_width, 3), dtype=np.float32)
    dest_mask = np.zeros((dest_height, dest_width), dtype=np.uint8)
    
 
    for y in range(dest_height):
        for x in range(dest_width):
            # Map the (x, y) coordinate in the destination image back to the source image
            dest_coords = np.array([x, y, 1])
            src_coords = destToSrc_H.dot(dest_coords)
            src_coords /= src_coords[2]
            
            src_x, src_y = src_coords[:2]
            if 0 <= src_x < src_img.shape[1] and 0 <= src_y < src_img.shape[0]:
                dest_img[y, x] = src_img[int(src_y), int(src_x)]
                dest_mask[y, x] = 1
                
# =============================================================================
#     dest_img_pil = Image.fromarray(dest_img)
#     dest_mask_pil = Image.fromarray(dest_mask * 255)
# =============================================================================
    
    return dest_img, dest_mask                
    raise NotImplementedError

def runRANSAC(src_pts: np.ndarray, dest_pts: np.ndarray, ransac_n: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
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
    max_inliers = 0
    best_homography = None
    best_inliers = np.array([])

    for i in range(ransac_n):
        idx = np.random.choice(src_pts.shape[0], size=4, replace=False)
        subset_src_pts = src_pts[idx]
        subset_dest_pts = dest_pts[idx]

        H = computeHomography(subset_src_pts, subset_dest_pts)
        
        transformed_pts = applyHomography(H, src_pts)

        distances = np.linalg.norm(transformed_pts - dest_pts, axis=1)

        inliers = np.where(distances < eps)[0]

        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_homography = H
            best_inliers = inliers

    return best_inliers, best_homography
    raise NotImplementedError

def blendImagePair(img1: List[np.ndarray], mask1: List[np.ndarray], img2: np.ndarray, mask2: np.ndarray, mode: str) -> np.ndarray:
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
    
    mask1 = mask1 / 255
    mask2 = mask2 / 255


    out_img = np.zeros_like(img1)
    
    if mode == 'overlay':
        blended_image = np.where(mask2[:, :, None] > 0, img2, img1)
    elif mode == 'blend':
        mask1 = distance_transform_edt(mask1)
        mask2 = distance_transform_edt(mask2)
        mask1[mask1==0] = 0.00000000001
        mask2[mask2==0] = 0.00000000001
        blended_image = (img1 * mask1[..., np.newaxis] + img2 * mask2[..., np.newaxis]) / (mask1 + mask2)[..., np.newaxis]

    out_img = blended_image.astype(np.uint8)
    return out_img
    raise NotImplementedError


def stitchImg(*args: Image.Image) -> Image.Image:
    '''
    Stitch a list of images.
    Arguments:
        args: a variable number of input images.
    Returns:
        stitched_img: the stitched image.
    '''
    imgs_np = args  # Convert PIL Images to numpy arrays and normalize

    start = imgs_np[0]  # Use the first image as the start

    for i in range(1, len(imgs_np)):
        
        curr = imgs_np[i]
        xs, xd = genSIFTMatches(curr, start)
        xs = xs[:, [1, 0]] 
        xd = xd[:, [1, 0]]
        np.random.seed(7777)
        _, H = runRANSAC(xs, xd, ransac_n=1000, eps=1)
        w, h = curr.shape[1], curr.shape[0]
        corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
# =============================================================================
#         i1 = Image.fromarray((start * 255).astype(np.uint8))
#         i2 = Image.fromarray((curr * 255).astype(np.uint8))
#         i1.show()
#         i2.show()
# =============================================================================
        warped_corners = applyHomography(H, corners)  
        max_width = max(int(np.ceil(np.max(warped_corners[:, 0]))), start.shape[1])
        max_height = max(int(np.ceil(np.max(warped_corners[:, 1]))), start.shape[0])
        
        canvas_shape = (max_height, max_width, 3)
        canvas = np.zeros(canvas_shape)
        canvas[:start.shape[0], :start.shape[1], :] = start
        
        canvas_mask = np.zeros(canvas_shape[:2])
        start_mask = np.ones((start.shape[0], start.shape[1]))
        canvas_mask[:start.shape[0], :start.shape[1]] = start_mask
# =============================================================================
#         i3 = Image.fromarray((canvas * 255).astype(np.uint8))
#         i4 = Image.fromarray((canvas_mask * 255).astype(np.uint8))
#         i3.show()
#         i4.show()
# =============================================================================
        warped_img, warped_mask = backwardWarpImg(curr, np.linalg.inv(H), canvas_shape[:2])
# =============================================================================
#         print(warped_mask.shape)
#         i5 = Image.fromarray((warped_img * 255).astype(np.uint8))
#         i6 = Image.fromarray((warped_mask * 255).astype(np.uint8))
#         i5.show()
#         i6.show()
# =============================================================================
        canvas_mask = canvas_mask.squeeze()
        blended_img = blendImagePair((canvas*255).astype(np.uint8), (canvas_mask*255).astype(np.uint8), (warped_img*255).astype(np.uint8), (warped_mask*255).astype(np.uint8), mode='blend')
# =============================================================================
#         i7 = Image.fromarray(blended_img.astype(np.uint8))
#         i7.show()
# =============================================================================
        start = blended_img /255.0
        
    final_img = Image.fromarray((blended_img).astype(np.uint8))
    return final_img
    raise NotImplementedError
    


