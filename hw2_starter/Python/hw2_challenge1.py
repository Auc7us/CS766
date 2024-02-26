import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import skimage
from PIL import Image
import math


def generateLabeledImage(gray_img: np.ndarray, threshold: float) -> np.ndarray:
    '''
    Generates a labeled image from a grayscale image by assigning unique labels to each connected component.
    Arguments:
        gray_img: grayscale image.
        threshold: threshold for the grayscale image.
    Returns:
        labeled_img: the labeled image.
    '''
    bin_img  = (gray_img > threshold).astype(np.uint8)
    labeled_img = skimage.measure.label(bin_img)
    # deb_img = labeled_img*50
    # matplotlib.pyplot.figure()
    # matplotlib.pyplot.imshow(deb_img)
    return labeled_img
    raise NotImplementedError

def compute2DProperties(orig_img: np.ndarray, labeled_img: np.ndarray) ->  np.ndarray:
    '''
    Compute the 2D properties of each object in labeled image.
    Arguments:
        orig_img: the original image.
        labeled_img: the labeled image.
    Returns:
        obj_db: the object database, where each row contains the properties
            of one object.
    '''
    obj_count = np.max(labeled_img)
    
    obj_db = np.zeros((obj_count,8))
    
    for i in range(obj_count):
        obj_i = labeled_img == i+1
        # obj_i = obj_i/(i+1)
        # matplotlib.pyplot.figure()
        # matplotlib.pyplot.imshow(obj_i)
        obj_i_ar = 0
        cx_obj_i = 0
        cy_obj_i = 0
        
        nRow = np.shape(obj_i)[0]
        nCol = np.shape(obj_i)[1]
        
        for j in range(nRow):
            for k in range(nCol):
                obj_i_ar = obj_i_ar + obj_i[j][k]
                cx_obj_i = cx_obj_i + k * obj_i[j][k] # x is cols
                cy_obj_i = cy_obj_i + j * obj_i[j][k] # y is rows
        
        cx_obj_i = cx_obj_i/obj_i_ar # x is cols
        cy_obj_i = cy_obj_i/obj_i_ar # y is rows
        
        temp_a = 0
        temp_b = 0
        temp_c = 0
        
        for j in range(nRow):
            for k in range(nCol):
                
                temp_c += (k-cx_obj_i)     * (k-cx_obj_i)   * obj_i[j][k]
                temp_b += 2 * (k-cx_obj_i) * (j-cy_obj_i)   * obj_i[j][k]
                temp_a += (j-cy_obj_i)     * (j-cy_obj_i)   * obj_i[j][k]
                

        ori = math.atan2(temp_b, temp_a-temp_c) / 2 
        
        e1 = temp_a * math.sin(ori)**2 - temp_b * math.sin(ori) * math.cos(ori) + temp_c * math.cos(ori)**2
        e2 = temp_a * math.sin(ori+math.pi/2)**2 - temp_b * math.sin(ori+math.pi/2) * math.cos(ori+math.pi/2) + temp_c * math.cos(ori+math.pi/2)**2
        
        eMin = e1
        if eMin > e2:
            eMin = e2
            eMax = e1
            ori += math.pi/2
        else:
            eMax = e2

        roundness = eMin/eMax
        
        
        obj_db[i][0] = i+1
        obj_db[i][1] = cx_obj_i
        obj_db[i][2] = cy_obj_i
        obj_db[i][3] = eMin
        obj_db[i][4] = (ori*180/math.pi)+90
        obj_db[i][5] = roundness
        obj_db[i][6] = obj_i_ar
        obj_db[i][7] = ori
        
        
    return obj_db
    raise NotImplementedError

def recognizeObjects(orig_img: np.ndarray, labeled_img: np.ndarray, obj_db: np.ndarray, output_fn: str):
    '''
    Recognize the objects in the labeled image and save recognized objects to output_fn
    Arguments:
        orig_img: the original image.
        labeled_img: the labeled image.
        obj_db: the object database, where each row contains the properties 
            of one object.
        output_fn: filename for saving output image with the objects recognized.
    '''
    test_db = compute2DProperties(orig_img,labeled_img)
    det_obj_arr = np.ones(len(obj_db), dtype=int)
    det_obj_arr = det_obj_arr * 255
    for i in range(len(obj_db)):
        for j in range(len(test_db)):
            v1 = obj_db[i][5] / test_db[j][5] # Roundness
            v2 = obj_db[i][3] / test_db[j][3] # Min Moment
            print("v1:",v1,"v2:",v2)
            if v1 < 1.10 and v1 > 0.9 and v2 < 1.2 and v2 > 0.8:
                det_obj_arr[i] = j 
    # print("#######",det_obj_arr)
    fig, ax = plt.subplots()
    plt.axis(False)
    ax.imshow(orig_img, cmap='gray')
    for i in det_obj_arr:
        if i < 255:
            ax.plot(test_db[i][1], test_db[i][2], 'b*', markerfacecolor='w')    
            length = 50
            endX = test_db[i][1] + length * math.sin(test_db[i][7])
            endY = test_db[i][2] + length * math.cos(test_db[i][7])
            x_lin = [test_db[i][1], endX]
            y_lin = [test_db[i][2], endY]
            plt.plot(x_lin, y_lin, color="red", linewidth=1)
    plt.savefig(output_fn)
    plt.show()
    return 0
    raise NotImplementedError


def hw2_challenge1a():
    import matplotlib.cm as cm
    from skimage.color import label2rgb
    from hw2_challenge1 import generateLabeledImage
    img_list = ['two_objects.png', 'many_objects_1.png', 'many_objects_2.png']
    threshold_list = [0.5, 0.5, 0.5]   # You need to find the right thresholds

    for i in range(len(img_list)):
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.
        labeled_img = generateLabeledImage(orig_img, threshold_list[i])
        Image.fromarray(labeled_img.astype(np.uint8)).save(
            f'outputs/labeled_{img_list[i]}')
        
        cmap = np.array(cm.get_cmap('Set1').colors)
        rgb_img = label2rgb(labeled_img, colors=cmap, bg_label=0)
        Image.fromarray((rgb_img * 255).astype(np.uint8)).save(
            f'outputs/rgb_labeled_{img_list[i]}')

def hw2_challenge1b():
    labeled_two_obj = Image.open('outputs/labeled_two_objects.png')
    labeled_two_obj = np.array(labeled_two_obj)
    orig_img = Image.open('data/two_objects.png')
    orig_img = np.array(orig_img.convert('L')) / 255.
    obj_db  = compute2DProperties(orig_img, labeled_two_obj)
    np.save('outputs/obj_db.npy', obj_db)
    print(obj_db)
    
    # TODO: Plot the position and orientation of the objects
    # Use a dot or star to annotate the position and a short line segment originating from the dot for orientation
    # Refer to demoTricksFun.py for examples to draw dots and lines. 
    fig, ax = plt.subplots()
    plt.axis(False)
    ax.imshow(orig_img, cmap='gray')
    for i in range(obj_db.shape[0]):
        ax.plot(obj_db[i][1], obj_db[i][2], 'b*', markerfacecolor='w')    
        length = 50
        endX = obj_db[i][1] + length * math.sin(obj_db[i][7])
        endY = obj_db[i][2] + length * math.cos(obj_db[i][7])
        x_lin = [obj_db[i][1], endX]
        y_lin = [obj_db[i][2], endY]
        plt.plot(x_lin, y_lin, color="red", linewidth=1)
    plt.savefig('outputs/two_objects_properties.png')
    plt.show()
    

def hw2_challenge1c():
    obj_db = np.load('outputs/obj_db.npy')
    img_list = ['many_objects_1.png', 'many_objects_2.png']

    for i in range(len(img_list)):
        labeled_img = Image.open(f'outputs/labeled_{img_list[i]}')
        labeled_img = np.array(labeled_img)
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.

        recognizeObjects(orig_img, labeled_img, obj_db, f'outputs/testing1c_{img_list[i]}')
    
    obj1_db = compute2DProperties(np.array(Image.open('data/many_objects_1.png')), np.array(Image.open('outputs/labeled_many_objects_1.png')))
    print(obj1_db)
    img1_list = ['two_objects.png', 'many_objects_2.png']    
    for i in range(len(img1_list)):
        labeled_img = Image.open(f'outputs/labeled_{img1_list[i]}')
        labeled_img = np.array(labeled_img)
        orig_img = Image.open(f"data/{img1_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.

        recognizeObjects(orig_img, labeled_img, obj1_db, f'outputs/testing1cwithMO1_{img1_list[i]}')


