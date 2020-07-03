import numpy as np
import pandas as pd
import skimage
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#tf.executing_eagerly()
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, io, color, exposure, img_as_float, transform, filters

import os
from flask import request, url_for, jsonify
from flask_api import FlaskAPI, status, exceptions
from skimage.color import rgb2gray

app = FlaskAPI(__name__)


def loadDataGeneral(df, path, im_shape):
    X, y = [], []
    for i, item in df.iterrows():
        #img = img_as_float(io.imread(path + item[0]))
        img = io.imread(item[0])
        print(img.shape)
        if len(img_as_float(img).shape)>3:
            img = rgba2rgb(img)
        img = rgb2gray(img)
        print(img.shape)
        img = img_as_float(img)
        #mask = io.imread(path + item[1])
        img = transform.resize(img, im_shape)
        print(img.shape)
        img = exposure.equalize_hist(img)
        print(img.shape)
        img = np.expand_dims(img, -1)
        print(img.shape)
        #mask = transform.resize(mask, im_shape)
        #mask = np.expand_dims(mask, -1)
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

def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
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

from pandas import DataFrame
from PIL import Image

#read image
# img_grey = cv2.imread('D:/original.png', cv2.IMREAD_GRAYSCALE)

# # define a threshold, 128 is the middle of black and white in grey scale
# thresh = 128

# # threshold the image
# img_binary = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)[1]

# #save image
# cv2.imwrite('D:/black-and-white.png',img_binary) 
# from PIL import Image
# def black_and_white(input_image_path,
#     output_image_path):
#    color_image = Image.open(input_image_path)
#    bw = color_image.convert('L')
#    bw.save(output_image_path)

# gray_img = skimage.color.rgb2gray(image)

#     print(hsv_img[:, :, 0])

#     thresh = skimage.filters.threshold_otsu(gray_img)
#     dst = (gray_img <= thresh) * 1.0
#     io.imsave('./image.jpg', dst)


#if __name__ == '__main__':
@app.route('/mask', methods=['GET'])
def get_mask():
    try:
        
        model_name = './trained_model.hdf5'
        print(model_name)
        UNet = load_model(model_name)
        #df = pd.read_csv(csv_path)
    
        fileName = request.args.get('fileName')
        workdir = request.args.get('workdir')
        
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

            
            #filenn = os.path.splitext(filen)[0]
            #filen = ntpath.basename(filelist[0])
            #plt.savefig(os.path.join(workdir, os.path.splitext(filen)[0] + "-mask" + ".jpg"))
            #path = os.path.join(workdir, os.path.splitext(filen)[0] + "-mask" + ".jpg")
            #plt.savefig(path)
            
            #cv2.imwrite(path, img_binary)
        
            i += 1
            if i == n_test:
                break
        return jsonify(status="success", fname = path), status.HTTP_200_OK
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        return jsonify(status='failure'), status.HTTP_500_INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    app.run(debug=True, port = 8000)
