"""
    app.py - Building a simple ResNet Keras deep learning REST api
    Author: Sadip Giri (sadipgiri@bennington.edu)
    Date: 23rd Aug. 2018
"""
import flask
import io
from keras.applications import ResNet50, imagenet_utils
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import tensorflow as tf

# initialize Flask application
app = flask.Flask(__name__)
# initialize Keras model
model = None

# Instantiating ResNet50 (pre-trained Keras model on the ImageNet dataset) architechture and loading weights from disk
def load_model():
    global model
    model = ResNet50(weights="imagenet")
    """
        Need to initialize global graph and assign it to tensorflow's default graph
        due to debugging issue with Flask
    """
    global graph
    graph = tf.get_default_graph()  

def preprocess_image(image, target):
    """
        Accept an input image
        Convert the mode to RGB (if necessary)
        Resize the image to 224x224 pixels (the input spatial dimensions for ResNet)
        Preprocess the array via mean subtraction and scaling
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image

@app.route('/predict', methods=["POST"])
def predict():
   """
    The data dictionary is used to indicate if prediction was successful or not and also to store the results of any predictions we make on the incoming data.
    To accept the incoming data we check if:
        The request method is POST (enabling us to send arbitrary data to the endpoint, including images, JSON, encoded-data, etc.)
        An image has been passed into the files attribute during the POST
    Then take the incoming data and:
        Read it in PIL format
        Preprocess it
        Pass it through our network
        Loop over the results and add them individually to the data["predictions"] list
        Return the response to the client in JSON format
   """
   data = {"success": False}
   if flask.request.method == "POST":
       if flask.request.files.get("image"):
           image = flask.request.files["image"].read()
           image = Image.open(io.BytesIO(image))
           image = preprocess_image(image, target=(224, 224))
           with graph.as_default():
                preds = model.predict(image)
                results = imagenet_utils.decode_predictions(preds)
                data["predictions"] = []
                for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)
                data["success"] = True   
   return flask.jsonify(data)

if __name__ == '__main__':
    print("* Loading Keras model and Flask server")
    load_model()
    print("Please wait until server has fully started")
    app.run(host='0.0.0.0', port=80)
    