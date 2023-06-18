from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os
import joblib


app = Flask(__name__)
def remove_img(self, path, img_name):
        if os.path.exists(path + '/' + img_name):
            os.remove(path + '/' + img_name) 
            # having a return value destroyed the API

def preprossing(recievedImage):
    remove_img(remove_img, 'static/IMG', 'usedimg.jpeg')
    print('opening image')
    rImage = Image.open(recievedImage)
    imagePath = 'static/IMG/usedimg.jpeg'
    rImage.save(imagePath) # this converts to the format i give
    image = cv2.imread(imagePath)
    input_image_resize = cv2.resize(image, (256,256))
    input_image_scaled = input_image_resize/255
    image_reshaped = np.reshape(input_image_scaled, [1,256,256,3])
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

model=load_model("model.h5")
cropyield = joblib.load('finalmodel.pkl')


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
        input_prediction = model.predict(image_arr)
        print("Model predicting ...")
        print(input_prediction[0])
        print(max(input_prediction[0]))
        input_pred_label = np.argmax(input_prediction)
        if max(input_prediction[0]) < 0.6 :
            prediction = 'please add plants picture '
        else:
            prediction = classes[input_pred_label]
        remove_img(remove_img, 'static/IMG', 'usedimg.jpeg')
        print("Image is being deleted")

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
        image_arr = preprossing(image)
        input_prediction = model.predict(image_arr)
        print("Model predicting ...")
        print(input_prediction[0])
        print(max(input_prediction[0]))
        input_pred_label = np.argmax(input_prediction)
        if max(input_prediction[0]) < 0.6 :
            prediction = 'please add plants picture '
        else:
            prediction = classes[input_pred_label]
        print(prediction)

        return render_template('index.html', prediction=prediction, image='static/IMG/usedimg.jpeg', appName="plant disease detection")
    else:
        return render_template('index.html',appName="plant disease detection")

########################################################

@app.route('/cropyield')
def cropYieldPage():
    return render_template('predictcropyield.html')

@app.route('/predictCropYieldApi', methods=["POST"])
def cropYieldApi():
    try:
        if ('AverageTemp' or 'AverageRainfall' or 'CropName' or 'CountryName' or 'PesticideTonnes') not in request.form:
            return jsonify("Please try again. Enter valid fields")

        AverageTemp = request.form['AverageTemp']
        AverageRainfall = request.form['AverageRainfall']
        CropName = request.form['CropName']
        CountryName = request.form['CountryName']
        PesticideTonnes = request.form['PesticideTonnes']
        data = np.array([[CountryName, CropName, AverageRainfall, PesticideTonnes, AverageTemp]])
        prediction = cropyield.predict(data)
        prediction = cropyield.predict(data)
        return jsonify({'prediction': float(prediction[0])})
    except:
        return jsonify({'Error': 'Error occur'})



@app.route('/predictCropYield', methods=["POST"])
def cropYield():
    if request.method == "POST":
        AverageTemp = request.form['AverageTemp']
        AverageRainfall = request.form['AverageRainfall']
        CropName = request.form['CropName']
        CountryName = request.form['CountryName']
        PesticideTonnes = request.form['PesticideTonnes']
        data = np.array([[CountryName, CropName, AverageRainfall, PesticideTonnes, AverageTemp]])
        prediction = cropyield.predict(data)
        print(prediction)
        return render_template('predictcropyield.html', prediction=prediction)
    else:
        print('not printed')
        return  render_template('predictcropyield.html')


if __name__ == '__main__':
    app.run(debug=True)