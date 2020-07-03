import numpy as np
from skimage import io
from skimage.feature.texture import local_binary_pattern
import matplotlib.pyplot as plt
from flask import request, url_for, jsonify
from flask_api import FlaskAPI, status, exceptions
from skimage import morphology, io, color, exposure, img_as_float, transform, filters
from skimage.color import rgb2gray

app = FlaskAPI(__name__)


@app.route('/lbp', methods=['GET'])
def get_vector():
    try:
        fileName = request.args.get('fileName')
        maskName = request.args.get('maskName')
        h_bins = np.arange(59)
        #h_range = (0, dim)
        P = 8
        R = 2
        
        img = io.imread(fileName)
        print(img.shape)
        img = rgb2gray(img)
        print(img.shape)
        mask = io.imread(maskName)
        print(mask.shape)
        img = transform.resize(img, mask.shape)
        print(img.shape)
        codes = local_binary_pattern(img, P, R)
        print("here")
        n_bins = int(codes.max() + 1)
        h_img, _ = np.histogram(codes.ravel(), density = False, bins=h_bins, range=(0, n_bins))
        h_masked, _ = np.histogram(codes[mask], density = False, bins=h_bins, range=(0, n_bins))
        return jsonify(status="success", lbp = (h_masked.tolist())), status.HTTP_200_OK
        #flask.jsonify(data)
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        return {'status':'failure'}, status.HTTP_500_INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    app.run(debug=True, port=4000)