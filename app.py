import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, request, render_template
from keras.applications.resnet_v2 import preprocess_input  

app = Flask(__name__)

classification_model = load_model("models/ResNet152V2.h5", compile=False)
identification_model = load_model("models/SnakeDetectionModel_ResNet50.h5", compile=False)

@app.route("/")
def index():
    return render_template("index.html")

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