#!/usr/bin/env python
# coding: utf-8

# In[25]:


import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os, json
import pandas as pd
from lime import lime_image
from skimage.segmentation import mark_boundaries

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import r

import tempfile
import ntpath

from skimage.io import imsave

from flask import request, url_for, jsonify
from flask_api import FlaskAPI, status, exceptions

from PIL import Image, ImageDraw, ImageFilter
from skimage import data
from matplotlib import pyplot as plt
from skimage import morphology, io, color, exposure, img_as_float, transform, filters
from skimage.color import rgb2gray, gray2rgb
from skimage.measure import compare_ssim
import skimage
from pandas import DataFrame
from PIL import Image
import cv2

app = FlaskAPI(__name__)

grdevices = importr('grDevices')


r = robjects.r
r.source('rpy__.R')
r_f = robjects.globalenv['do_prediction_with_mask']
print(r_f)


import numpy as np
from skimage import io
from skimage.feature.texture import local_binary_pattern
import matplotlib.pyplot as plt
from flask import request, url_for, jsonify
from flask_api import FlaskAPI, status, exceptions
from skimage import morphology, io, color, exposure, img_as_float, transform, filters
from skimage.color import rgb2gray

app = FlaskAPI(__name__)


class explainer_cls:
    def __init__(self, org_img, org_mask, org_image_name):
        self.org_img = org_img
        self.org_mask = org_mask
        self.org_image_name = org_image_name
    
    def get_new_img(self, n_name, img):
        per = io.imread(img)
        print(per.shape)
        #per = rgb2gray(per)
        #per = transform.resize(per, self.org_mask.shape)
        (score, diff) = compare_ssim(self.org_img, per, full=True, multichannel=True)
        diff = (diff * 255).astype("uint8")
        #thresh = skimage.filters.threshold_otsu(diff)
        #thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #dst = (diff >= thresh) * 255.0
        diff = rgb2gray(diff)
        dd = (diff>0)*self.org_mask
        #pp = self.org_img.copy()*(dd>0)
        io.imsave(n_name, dd)
        

    def batch_predict(self, images):
        print(images.shape)
        ret = None
        for i in range(0, len(images)):
            k = images[i]
            im = Image.fromarray(k, 'RGB')
            tf = tempfile.NamedTemporaryFile()
            name = tf.name+".png"
            im.save(name)
            tf = tempfile.NamedTemporaryFile()
            name1 = tf.name+".png"
            self.get_new_img(name1, name)
            pf_dt = robjects.conversion.rpy2py(r_f(self.org_image_name, name1, name))
            x, y = pf_dt
            
            data = np.array([x[0], y[0]])
            print(data)
            if ret is None:
                ret = [data]
            else:
                ret = np.vstack((ret, data))
            os.unlink(name)
            os.unlink(name1)
        print(ret.shape)
        return ret




    def get_image(self, path):
        with open(os.path.abspath(path), 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB') 




    def createExplainer(self, img):
        explainer = lime_image.LimeImageExplainer()
        img = self.get_image(img)
        explanation = explainer.explain_instance(np.array(img), self.batch_predict, top_labels=5, hide_color=0, num_samples=100)
        return explanation




    def dotesting(self, explanation):
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        img_boundry1 = mark_boundaries(temp/255.0, mask)
        return img_boundry1





import numpy as np
import pandas as pd
import skimage
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, io, color, exposure, img_as_float, transform, filters

import os
from flask import request, url_for, jsonify
from flask_api import FlaskAPI, status, exceptions
from skimage.color import rgb2gray

def loadDataGeneral(df, path, im_shape):
    X, y = [], []
    for i, item in df.iterrows():
        img = io.imread(item[0])
        print(img.shape)
        if len(img_as_float(img).shape)>3:
            img = rgba2rgb(img)
        img = rgb2gray(img)
        print(img.shape)
        img = img_as_float(img)
        img = transform.resize(img, im_shape)
        print(img.shape)
        img = exposure.equalize_hist(img)
        print(img.shape)
        img = np.expand_dims(img, -1)
        print(img.shape)
        X.append(img)
        #y.append(mask)
    X = np.array(X)
    print(X.shape)
    #y = np.array(y)
    X -= X.mean()
    X /= X.std()

    print ('### Dataset loaded')
    print ('\t{}'.format(path))
    print ('\t{}'.format(X.shape))
    print ('\tX:{:.1f}-{:.1f}\t\n'.format(X.min(), X.max()))
    print ('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
    return X

def masked(img, gt, mask, alpha=1):
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    boundary = morphology.dilation(gt, morphology.disk(3)) ^ gt
    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img




def get_mask(fileName, workdir):

    
    model_name = './trained_model.hdf5'
    print(model_name)
    UNet = load_model(model_name)
    
    filelist = [fileName]
    filen = os.path.basename(filelist[0])
    path = os.path.join(workdir, os.path.splitext(filen)[0] + "-mask" + ".png")
    




    df = DataFrame (filelist,columns=['file'])
    print (df)


    # Load test data
    im_shape = (256, 256)
    X = loadDataGeneral(df, filelist[0], im_shape)
    print(X.shape)
    n_test = X.shape[0]
    inp_shape = X[0].shape
    
    # Load model

    
    print("model loaded")
    # For inference standard keras ImageGenerator can be used.
    test_gen = ImageDataGenerator(rescale=1.)
    print("here")
    

    gts, prs = [], []
    i = 0
    for xx in test_gen.flow(X, batch_size=1):
        print(xx.shape)
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
        pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
        #mask = yy[..., 0].reshape(inp_shape[:2])

        #gt = mask > 0.5
        pr = pred > 0.5

        pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
    

        #plt.imshow(pred, cmap='gray')
        gray_img = skimage.color.rgb2gray(pred)
        thresh = skimage.filters.threshold_otsu(gray_img)
        dst = (gray_img >= thresh) * 255.0
        print(np.unique(dst))
        #bw = im.convert('L')
        io.imsave(path, dst)

        
    
        i += 1
        if i == n_test:
            break
    return path





##########################################################################################


def controller(image, exp):
    img = image
    workingDir = exp
    fname = ntpath.basename(img)
    tempName = ntpath.join(workingDir, fname.replace(".jpg", ".png"))
    tempName_for_explainer = ntpath.join(workingDir, fname+"tmp.png")
    
    mask = get_mask(image, exp)
    mask = io.imread(mask)
    org = io.imread(image)
    org = transform.resize(org, mask.shape)
    if len(org.shape)>2:
        org = rgb2gray(org)
        org = skimage.img_as_ubyte(org)
    k = (mask>0)*org
    k = gray2rgb(k)
    
    org = gray2rgb(org)
    
    #print(np.unique(k))
    print("**************************************")
    io.imsave(tempName_for_explainer, k)
    print("**************************************")
    explain_obj = explainer_cls(org, mask, img)
    
    
    exp = explain_obj.createExplainer(tempName_for_explainer)
    imb = explain_obj.dotesting(exp)
    imsave(tempName, imb)


    
    

    
if __name__ == '__main__':
    controller("D:\\dke\\2ND SEM\\R Project\\resource\\covid-dataset\\covid-dataset\\covid-negative\\4.jpg", "D:\\dke\\2ND SEM\\R Project\\app\\explain server\\wd")
    
    







################################################################
