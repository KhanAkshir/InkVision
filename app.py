import numpy as np
from flask import Flask,request,render_template
import pandas as pd
from PIL import Image
import warnings

import tensorflow as tf
model_clf = tf.keras.models.load_model("num_model.h5")

warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/',methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method == 'POST':
        # Get the uploaded image file from the form
        image_file = request.files['image']

        # print(image_file)
        
        # Open the image using PIL
        image = Image.open(image_file)
        resized_image = image.resize((28,28))
        # print(resized_image)
        # Convert the image to a numpy array
        image_array = np.array(resized_image)[:,:,0]

        # print(image_array)
        # print(type(image_array))
        # print(image_array)
        img=np.invert(np.array([image_array]))
        y_pred=model_clf.predict(img)
        # print(y_pred)
        ans=np.argmax(y_pred)
       

        # Process the image array as needed (e.g., perform calculations, apply filters, etc.)

        # Return the processed image or perform further actions
        return render_template('home.html',result=ans)
    else:
        return render_template('home.html',result=0)
      

if __name__=="__main__":
    app.run(host="0.0.0.0" )