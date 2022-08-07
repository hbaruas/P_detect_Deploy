from crypt import methods
from flask import Flask, render_template,request


import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras_preprocessing import image
import numpy as np

model = load_model('./Model/manual.hdf5')

app = Flask(__name__)

@app.route('/',methods=["GET"])

def hello_word():
    return render_template('index.html')


@app.route('/',methods= ['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = './static/' + imagefile.filename
    imagefile.save(image_path)

    test_image = load_img(image_path, target_size=(180, 180))
    test_image = img_to_array(test_image)
    test_image /= 255
    test_image = np.expand_dims(test_image, axis=0)   
    
    prediction = model.predict(test_image)
    prediction1 = prediction[0][0] * 100

    prediction1 = round(prediction1,2)

    


    return render_template('index.html',predict = prediction1, image = image_path )

if __name__ == "__main__":
    #app.run(port = 3000,debug = True)
    app.run(debug = True)
