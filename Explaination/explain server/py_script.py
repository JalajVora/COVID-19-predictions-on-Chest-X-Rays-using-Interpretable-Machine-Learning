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

app = FlaskAPI(__name__)

grdevices = importr('grDevices')


r = robjects.r
r.source('rpy__.R')
r_f = robjects.globalenv['do_prediction']
print(r_f)



def batch_predict(images):
    print(images.shape)
    ret = None
    for i in range(0, len(images)):
        k = images[i]
        im = Image.fromarray(k, 'RGB')
        tf = tempfile.NamedTemporaryFile()
        name = tf.name+".png"
        im.save(name)
        pf_dt = robjects.conversion.rpy2py(r_f(name))
        x, y = pf_dt
        
        data = np.array([x[0], y[0]])
        print(data)
        if ret is None:
            ret = [data]
        else:
            ret = np.vstack((ret, data))
        os.unlink(name)
    print(ret.shape)
    return ret




def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 




# In[ ]:
#explanation
def createExplainer(img):
    explainer = lime_image.LimeImageExplainer()
    img = get_image(img)
    explanation = explainer.explain_instance(np.array(img), batch_predict, top_labels=5, hide_color=127, num_samples=10)
    return explanation




def dotesting(explanation):
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    return img_boundry1




@app.route('/explain', methods=['GET'])
def controller():

    try:
        img = request.args.get('image')
        workingDir = request.args.get('exp')
        fname = ntpath.basename(img)
        tempName = ntpath.join(workingDir, fname)
        exp = createExplainer(img)
        imb = dotesting(exp)
        #print(imb)
        #plt.imshow(imb)
        imsave(tempName, imb)
        return jsonify(name=tempName), status.HTTP_200_OK
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        return {'status':'failure'}, status.HTTP_500_INTERNAL_SERVER_ERROR


    
    

    
if __name__ == '__main__':
    app.run(debug=True)
