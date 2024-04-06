import numpy as np
import os
import cv2
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, request, render_template
from keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

classification_model = load_model("models/ResNet152V2.h5", compile=False)
identification_model = load_model("models/SnakeDetectionModel_ResNet50.h5", compile=False)

# load model
identification_model = load_model('/Users/new/Downloads/SnakeDetectionModel_ResNet50.h5')
classification_model = load_model('/Users/new/Downloads/ResNet152V2.h5', compile = False)

@app.route("/")
def index():
    return render_template("index.html")


# Image path
image_path_folder = '/Users/new/Downloads/ratsnake-basking-on-forest_med_hr.jpeg'
nisith_img = '/Users/new/Downloads/ratsnake-basking-on-forest_med_hr.jpeg'


# Component 1 image preprocess
def identification_image_preprocess(image_path_folder):
    img = cv2.imread(image_path_folder)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    return img


# Component 2 image preprocess
def classification_image_preprocess(image_path):
    img = image.load_img(nisith_img, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# Component 1 model
nina_model_preprocess_image = identification_image_preprocess(nisith_img)
predictions = identification_model.predict(nina_model_preprocess_image)
print(predictions)
predicted_class = np.argmax(predictions)
print(f"Predicted Class Nina: {predicted_class}")

# Component 2 model
if predicted_class == 1:
    print("Nisitha")
    nisitha_image = classification_image_preprocess(image_path_folder)

    y = classification_model.predict(nisitha_image)
    preds = np.argmax(y)
    index = ["Common Rat Snake", "Russel's Viper", "Forsten's Cat Snake", "Green Pit Viper"]
    print("The classified Animal is : " + str(index[preds]))
    print("predictions", y)


def is_snake(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    print(x)
    x = np.expand_dims(x, axis=0)
    print(x)
    x = preprocess_input(x)
    y = identification_model.predict(x)
    print("predictions", y)
    return y[0][1] > 0.5


@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files["image"]
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath, "uploads", f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)

        if is_snake(filepath):
            img = image.load_img(filepath, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            y = classification_model.predict(x)
            preds = np.argmax(y)
            index = ["Common Rat Snake", "Russel's Viper", "Forsten's Cat Snake", "Green Pit Viper"]
            text = "The classified Animal is : " + str(index[preds])
            return render_template("index.html", pred_text=text)
        else:
            return render_template("index.html", pred_text="This is not a snake.")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False, threaded=False)