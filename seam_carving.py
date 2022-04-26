from typing import Dict, Any
import utils
NDArray = Any
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta



def resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    """

    :param image: ِnp.array which represents an image.
    :param out_height: the resized image height
    :param out_width: the resized image width
    :param forward_implementation: a boolean flag that indicates whether forward or basic implementation is used.
                                    if forward_implementation is true then the forward-looking energy is used otherwise
                                    the basic implementation is used.
    :return: A dictionary with three elements, {'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3},
            where img1 is the resized image and img2/img3 are the visualization images
            (where the chosen seams are colored red and black for vertical and horizontal seams, respecitvely).
    """
    start = timer()
    gsImg = utils.to_grayscale(image)

    k_h, k_w, increase_h, increase_w, decrease_h, decrease_w = init(gsImg, out_height, out_width)

    vertical_seams = image

    resized = image

    if increase_w or decrease_w:
        booleanMat_w = change_size(gsImg, k_w , out_height, out_width, forward_implementation)
        vertical_seams = final_results(image.copy(), booleanMat_w, "red", k_w, None)
        if decrease_w:
            resized = final_results(image.copy(), booleanMat_w, "delete", k_w, 'decrease_w')
        else:
            resized = final_results(image.copy(), booleanMat_w, "add", k_w, 'increase_w')
    horizontal_seams = resized
    if increase_h or decrease_h:
        resized = np.rot90(resized,k=1,axes=(0,1))
        booleanMat_h = change_size(utils.to_grayscale(resized), k_h, out_height, out_width, forward_implementation)
        horizontal_seams = final_results(resized.copy(), booleanMat_h, "black", k_h, None)
        horizontal_seams = np.rot90(horizontal_seams,k=-1,axes=(0,1))
        if decrease_h:
            #print("boolean shape is ", booleanMat_h.shape)
            #print("resized shape is ", resized.shape)
            resized = final_results(resized.copy(), booleanMat_h, "delete",k_h, 'decrease_h')
        else:
            resized = final_results(resized.copy(), booleanMat_h, "add", k_h, 'increase_h')
        resized = np.rot90(resized,k=-1,axes=(0,1))

    end = timer()
    print(timedelta(seconds=end - start))

    return {'resized' : resized, 'vertical_seams' : vertical_seams ,'horizontal_seams' : horizontal_seams}

def change_size(gsImg, k_w , out_height, out_width, forward_implementation):
    h = int(gsImg.shape[0]) # origin_height
    w = int(gsImg.shape[1]) # origin_width
    #print("h is : ",h, "w is : ",w)

    indexMat = indexMatrix(h, w) #build mat with all indexes

    energyMat = utils.get_gradients(gsImg)
    cv = np.zeros((h, w))
    cl = np.zeros((h, w))
    cr = np.zeros((h, w))

    if forward_implementation:
        cv, cl, cr = computeCs(gsImg, cv, cl, cr)  # otherwise zeros

    for i in range(k_w):

        M = energyMat.copy()
        #print("start cost matrix")
        M = costMatrix(M,cv, cl, cr, h, w-i)
        #print("finish cost matrix")
        #print("start finiding seam")
        indexMat, energyMat = findingSeam(M,energyMat, cv, cl,cr, indexMat) # find seam to remove and return all index that not deleted
        #print("finish finiding seam")
        #print("start build bool")
        booleanMat = buildBool(indexMat,h,w,w-i-1) # 0 - deleted 1 - keep

        #print("finish build bool")

        tmpGSImg = gsImg[booleanMat].reshape((h, w - i-1))  # Delete all the pixels marked 0 in the booleanMat, resiSze it to the new dim

        #energyMat = utils.get_gradients(tmpGSImg)

        cv = np.zeros((h, w-i-1))
        cl = np.zeros((h, w-i-1))
        cr = np.zeros((h, w-i-1))

        if forward_implementation:
            cv, cl, cr = computeCs(tmpGSImg, cv, cl, cr)  # otherwise zeros

    #utils.save_images({"gs":tmpGSImg}, "C:/Users/Liri/Desktop/graphic1/1", )

    return booleanMat


def init(image: NDArray, out_height: int, out_width: int):
    h, w = image.shape #original height and width
    k_h = 0
    k_w = 0
    increase_h = False
    decrease_h = False
    increase_w = False
    decrease_w = False
    if out_height > h:
        k_h = out_height - h
        increase_h = True
    elif out_height <  h:
        k_h = h - out_height
        decrease_h = True

    if out_width > w:
        k_w = out_width - w
        increase_w = True
    elif out_width < w:
        k_w = w - out_width
        decrease_w = True

    return k_h, k_w, increase_h, increase_w, decrease_h, decrease_w

def costMatrix(M, cv, cl, cr, h, w):
    m = np.zeros((h, w))
    for i in range(1, h):
        mU = M[i - 1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        j = 0
        while j < w:
            if j == 0:
                mULR = np.array([mU, mR])
                cULR = np.array([cv[i], cr[i]])
                mULR = np.add(mULR, cULR)
                argmins = np.argmin(mULR[:, [0]], axis=0)
                m[i][0] = np.choose(argmins, mULR[:, [0]])
                M[i][0] = np.add(M[i][0], m[i][0])
                j += 1
            elif j == w - 1:
                mULR = np.array([mU, mL])
                cULR = np.array([cv[i], cl[i]])
                mULR = np.add(mULR, cULR)
                argmins = np.argmin(mULR[:, [w - 1]], axis=0)
                m[i][w - 1] = np.choose(argmins, mULR[:, [w - 1]])
                M[i][w - 1] = np.add(M[i][w - 1], m[i][w - 1])
                j += 1
            else:
                mULR = np.array([mU, mL, mR])
                cULR = np.array([cv[i], cl[i], cr[i]])
                mULR = np.add(mULR, cULR)
                tmpMulr = mULR[:, np.r_[1:w - 1]]
                argmins = np.argmin(tmpMulr, axis=0)
                m[i, 1:w - 1] = np.choose(argmins, tmpMulr)
                M[i, 1:w - 1] += m[i, 1:w - 1]
                j = w - 1

    return M

def computeCs(gsImg, cv, cl, cr):
    zero_col = np.broadcast_to([0.], [gsImg.shape[0], 1])
    zero_raw = np.broadcast_to([0.], [1, gsImg.shape[1]])
    left = np.concatenate([zero_col, gsImg[:, 0:-1]], axis=1)
    right = np.concatenate([gsImg[:, 1:], zero_col], axis=1)
    top = np.concatenate([zero_raw, gsImg[:-1:]], axis=0)
    cv = np.abs(left - right)
    cl = np.abs(left - top) + cv
    cr = np.abs(right - top) + cv

    return cv, cl, cr

def indexMatrix(h,w):
    indexMat = np.zeros((h,w,2))
    for i in range(h):
        for j in range(w):
            indexMat[i,j] = (int(i),int(j))
    return indexMat

def findingSeam(M,energy, cv, cl,cr, indexMat):
    #print(cv)
    h, w = M.shape

    energy_copy = energy.copy()
    arr = list(M[h - 1])
    j = arr.index(min(arr))
    #print("M - ",M.shape, "energy - ", energy.shape, "cv - ", cv.shape)
    for i in range(h-1,0,-1):
        raw_up = {}
        x_v = energy[i, j] + M[i - 1, j] + cv[i, j] - M[i, j]
        raw_up[j] = x_v
        if j!= 0:
            x_l = energy[i,j]+ M[i-1,j-1] + cl[i,j] - M[i,j]
            raw_up[j-1] = x_l
        if j!= w-1:
            x_r = energy[i,j]+ M[i-1,j+1] + cr[i,j] - M[i,j]
            raw_up[j+1] = x_r
        minx = min(raw_up, key=raw_up.__getitem__)
        indexMat[i:i+1, j:-1] = indexMat[i:i+1, j + 1:]
        energy_copy[i:i+1, j:-1] = energy_copy[i:i+1, j + 1:]
        j = minx
    indexMat[0:1, j:-1] = indexMat[0:1, j + 1:] # for first line, which is outside the loop
    energy_copy[0:1, j:-1] = energy_copy[0:1, j + 1:] # for first line, which is outside the loop

    indexMat = np.delete(indexMat, w-1, 1)
    energy_copy = np.delete(energy_copy, w-1, 1)

    return indexMat, energy_copy

def buildBool(indexMat, h, w, w_index):
    indexMat = indexMat.astype(int)
    zeros = np.zeros((h,w), dtype=np.bool)
    row = (indexMat[:, :, 0])
    cols = (indexMat[:, :, 1])
    zeros[row, cols] = 1
    return zeros


def final_results(image, booleanMat, cond, k, decrease_or_increase):
    h, w = booleanMat.shape
    booleanMat_3d = np.stack((booleanMat, booleanMat, booleanMat), axis=2)
    #print(cond, "now we are = ",decrease_or_increase)
    if cond == 'delete':
        image = image[booleanMat_3d].reshape((h, w - k, 3))
    else:
        for i in range(h):
            for j in range(w):
                if booleanMat[i,j] == 0:
                    if cond == 'red':
                        image[i,j] = (255,0,0)
                    elif cond == 'black':
                        image[i, j] = (0, 0, 0)

    if cond == 'add':
        image = increase_w(image, booleanMat, h, w, k)

    return image

def increase_w(image, booleanMat, h, w, k):
    new_image = np.zeros((h,w+k,3))
    for i in range(h):
        sum = 0
        for j in range(w):
            #print(i,",",j)
            if booleanMat[i, j] == 1:
                new_image[i, j+sum] = image[i, j]
            if booleanMat[i, j] == 0:
                new_image[i, j+sum] = image[i, j]
                sum += 1
                new_image[i, j + sum] = image[i, j]

    return new_image

