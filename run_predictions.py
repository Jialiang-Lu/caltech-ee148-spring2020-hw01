import os
import numpy as np
import json
from PIL import Image

class DisjointSet:
    '''
    a simple implementation of a disjoint-set data structure
    '''
    _data = list()

    def __init__(self, init_data=None):
        self._data = []
        if init_data:
            for item in list(set(init_data)):
                self._data.append({item})

    def __repr__(self):
        return self._data.__repr__()
    
    def index(self, elem):
        for item in self._data:
            if elem in item:
                return self._data.index(item)
        return None

    def find(self, elem):
        for item in self._data:
            if elem in item:
                return self._data[self._data.index(item)]
        return None
    
    def add(self, elem):
        index_elem = self.index(elem)
        if index_elem is None:
            self._data.append({elem})
    
    def union(self, elem1, elem2):
        index_elem1 = self.index(elem1)
        index_elem2 = self.index(elem2)
        if index_elem1 is None:
            self.add(elem1)
            self.union(elem1, elem2)
            return
        if index_elem2 is None:
            self.add(elem2)
            self.union(elem1, elem2)
            return
        if index_elem1 != index_elem2:
            self._data[index_elem2] = self._data[index_elem2].union(self._data[index_elem1])
            del self._data[index_elem1]
        
    def get(self):
        return self._data

def label_binary(im):
    '''
    label each connected patch within a binary image
    '''
    if not isinstance(im, np.ndarray):
        im = np.asarray(im)
    im = (im>0).astype(int)
    labels = np.zeros_like(im)
    n_labels = 0
    idc = DisjointSet()
    for r, c in np.ndindex(im.shape):
        v = im[r, c]
        vu = labels[r-1, c] if r>0 else 0
        vl = labels[r, c-1] if c>0 else 0
        if v>0:
            if vu==0 and vl==0:
                n_labels += 1
                idc.add(n_labels)
                labels[r, c] = n_labels
            elif vu==0 and vl>0:
                labels[r, c] = vl
            elif vu>0 and vl==0:
                labels[r, c] = vu
            else:
                labels[r, c] = vu if vu<vl else vl
                idc.union(vu, vl)
    for r, c in np.ndindex(im.shape):
        v = labels[r, c]
        labels[r, c] = 0 if v==0 else idc.index(v)+1
    return labels, len(idc.get())

def apply_over_labels(im, labels, func):
    '''
    apply function over the connected labels labeled by labels
    '''
    return [func(labels==x) for x in range(1, labels.max()+1)]

def perimeter(im):
    '''
    calculate perimeter of binary image
    '''
    im = (im>0).astype(int)
    ima = np.zeros((im.shape[0]+2, im.shape[1]+2), dtype=int)
    ima[1:-1, 1:-1] = im
    imf = im*4-ima[1:-1, 2:]-ima[1:-1, :-2]-ima[2:, 1:-1]-ima[:-2, 1:-1]
    return (np.logical_and(imf>0, imf<4)).sum()

def area(im):
    '''
    calculate area of binary image
    '''
    return (im>0).sum()

def aspect_ratio(im):
    '''
    calculate aspect ratio of binary image
    '''
    return perimeter(im)**2/(4*np.pi*area(im))
    
def bounding_box(im):
    '''
    find bounding box of the binary image im
    '''
    idc = np.argwhere(im>0)
    return list(idc.min(axis=0))+list(idc.max(axis=0))

def bounding_boxes(labels):
    '''
    find all bounding boxes of the labels
    '''
    return [bounding_box(labels==v) for v in range(1, labels.max()+1)]

def circ_2d(ny, nx, radius):
    '''
    generate 2d circular kernel
    '''
    y, x = np.ogrid[-(ny-1)/2:(ny-1)/2:ny*1j, -(nx-1)/2:(nx-1)/2:nx*1j]
    kernel = ((y**2+x**2)<=radius**2).astype(float)
    kernel = kernel/kernel.sum()
    return kernel

def gaus_2d(ny, nx, sigma):
    '''
    generate 2d gaussian kernel
    '''
    y, x = np.ogrid[-(ny-1)/2:(ny-1)/2:ny*1j, -(nx-1)/2:(nx-1)/2:nx*1j]
    kernel = np.exp(-(y**2+x**2)/(2*sigma**2))
    kernel = kernel/kernel.sum()
    return kernel
    
def corr_2d(im, kernel):
    '''
    2d cross-correlation using fft
    '''
    fr = np.fft.fft2(im)
    fr2 = np.fft.fft2(kernel)
    m, n = fr.shape
    corr = np.real(np.fft.ifft2(fr*fr2))
    corr = np.roll(corr, -m//2, axis=0)
    corr = np.roll(corr, -n//2, axis=1)
    return corr

def gaus_filt_2d(im, sigma):
    '''
    2d gaussian filtering
    '''
    return corr_2d(im, gaus_2d(im.shape[0], im.shape[1], sigma))

def circ_filt_2d(im, radius):
    '''
    2d circular filtering
    '''
    return corr_2d(im, circ_2d(im.shape[0], im.shape[1], radius))

def remove_holes(im, labels, max_area=500):
    '''
    remove holes in binary image
    '''
    fill = True if im.dtype==np.dtype('bool') else 1
    for k in range(1, labels.max()+1):
        patch = labels==k
        if patch.sum()<max_area:
            im[patch] = fill

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    # Parameters
    radius_circ_kernel = 4 # radius of the circular step kernel
    sigma_gaussian_kernel = 1 # radius of the gaussian smooth kernel
    thr = 0.18 # threshold for detection
    area_min = 12 # mininum area of a patch to be considered as traffic light
    area_max = 1500 # maximum area of a patch to be considered as traffic light
    ratio_max = 0.85 # maximum aspect ratio of a patch to be considered as traffic light

    # Correlation and thresholding
    I_float = I.astype(float)/255
    I_single = I_float[:, :, 0]-I_float[:, :, 2]
    I_filt = circ_filt_2d(gaus_filt_2d(I_single, sigma_gaussian_kernel), radius_circ_kernel)
    I_thr = I_filt>thr

    # Labeling and locating bounding boxes
    labels, n_labels = label_binary(I_thr)
    labels_bg, n_labels_bg = label_binary(np.logical_not(I_thr))
    remove_holes(I_thr, labels_bg, 500)
    labels, n_labels = label_binary(I_thr)
    boxes = bounding_boxes(labels)
    areas = apply_over_labels(I_thr, labels, area)
    ratios = apply_over_labels(I_thr, labels, aspect_ratio)
    idc = [x[0] for x in np.argwhere(np.logical_and(np.logical_and(np.array(areas)>=area_min, np.array(areas)<area_max), np.array(ratios)<ratio_max))]
    boxes_final = np.array(boxes)[idc].tolist()

    for i in range(len(boxes_final)):
        assert len(boxes_final[i]) == 4
    
    return boxes_final

# set the path to the downloaded data: 
data_path = './data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = './data/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
for i in range(len(file_names)):
    
    print(f'Processing {os.path.join(data_path,file_names[i])}')
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
