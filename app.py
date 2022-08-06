# %%
# from werkzeug.wrappers import Request, Response
from pathlib import Path
from flask import Flask,send_file, request,jsonify
from object_detection import run_detection,detect_object_and_draw_decision_visulaization
import os
from flask_cors import CORS,cross_origin
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import json
import random
from PIL import Image 
import PIL 
app = Flask(__name__,static_folder="/static")

CORS(app,origins='*')
UPLOAD_FOLDER = '/path/to/the/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
port=5000


@app.route("/")
@cross_origin(origins='*')
def hello():
    return 'Hello flask!!!'


  
@app.route('/download/<file_name>',methods=[ 'GET'])
@cross_origin(origins='*')
def get_file(file_name):
    print(file_name)
    path=os.path.join(os.getcwd(),'static','detection', file_name)
    return send_file(path)

# if __name__ == '__main__':
#     app.run(host="localhost", port=port, debug=True)

def saveImageFile(file,path):
   print('CWD :{}'.format(os.getcwd()))
   if os.path.exists(path):
          print('FILE :{} EXISTS'.format(path))
   else:
          print('FILE :{} doen not EXIST'.format(path))
          cv2.imwrite(path, file)


@app.route('/cam',methods=[ 'POST'])
@cross_origin(origins='*')
def cam_file():
      file = request.files['file']
      print(type(file))
      filename:str = secure_filename(file.filename)
      path=os.path.join('static', filename)
      file.save(path)
      
      try:
        bbb= np.array(cv2.imread(path))
        image,image_with_cam=detect_object_and_draw_decision_visulaization(bbb)
        
        dynmicFileName='N_s{}{}'.format(random.randint(1,1000),filename)
        dynmicFileName_cam='M_s{}{}'.format(random.randint(1,1000),filename)
        dynmicName_path_detection=os.path.join(os.getcwd(),'static','detection', dynmicFileName)
        dynmicName_path_detection_cam=os.path.join(os.getcwd(),'static','detection', dynmicFileName_cam)

        saveImageFile(image,dynmicName_path_detection)
        saveImageFile(image_with_cam,dynmicName_path_detection_cam)
        return jsonify(filename=dynmicFileName,filename_cam=dynmicFileName_cam) 
      except:
        return  "Internal Error", 400


# %%
