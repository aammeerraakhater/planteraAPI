from flask import Flask, request, render_template, url_for, jsonify
from keras.applications.mobilenet import MobileNet
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import imghdr

app = Flask(__name__)

def preprossing(recievedImage):
    print('opening image')
    rImage = Image.open(recievedImage)
    imagePath = 'static/IMG/usedimg.jpeg'
    # rImage.save(imageFakePath)
    # imagetype = imghdr.what(imageFakePath)
    # imagePath = 'static/IMG/usedimg.'+imagetype # this is definitely useless
    rImage.save(imagePath) # it's convert to the format i give
    print('done')
    print('image name')
    image = cv2.imread(imagePath)
    print("image preprocsessing read the image")
    image_resized = cv2.resize(image,(224, 224))
    image_scaled = image_resized/255
    image_reshaped = np.reshape(image_scaled,[1,224,224,3])
    print("image preprocsessing returning the image")

    return image_reshaped


classes = ['Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)__Common_rust',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape__Esca(Black_Measles)',
        'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange__Haunglongbing(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,bell__Bacterial_spot',
        'Pepper,bell__healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy']
model=load_model("nn.h5")

@app.route('/')
def index():
    return render_template('index.html', appName="plant disease detection")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprossing(image)
        print("Model predicting ...")
        result = model.predict(image_arr)
        print("Model predicted")
        ind = np.argmax(result)
        prediction = classes[ind]
        print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        image_arr= preprossing(image)
        print(image_arr)
        print("predicting ...")

        result = model.predict(image_arr)
        print("predicted ...")
        ind = np.argmax(result)
        prediction = classes[ind]

        print(prediction)

        return render_template('index.html', prediction='prediction', image='static/IMG/', appName="plant disease detection")
    else:
        return render_template('index.html',appName="plant disease detection")


if __name__ == '__main__':
    app.run(debug=True)