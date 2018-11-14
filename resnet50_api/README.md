# Project Description

Building a RESTful API using Flask to classify certain image using pre-trained ResNet50 Keras Model.

# Framework and Libraries used:

- Docker
- Flask
- keras==2.1.6
- numpy
- Pillow
- tensorflow==1.8.0

# Getting Started Guide

Run Docker on your local machine

Go to the root of project directory using command line 
Build the docker image for the api using docker build -t “give_name e.g. resnet50api” .
After the build is successful, run the docker conainer using docker run -p 4000:80 "given_name e.g. resenet50api"

Fire up another commandline shell:
Go to the root of the project directory and connect to the port 4000 where Flask app is running and post the image using curl -X POST -F image=@filename 'http://localhost:4000/predict'

filename should be substituted by your own images or could use image test files which are in the sample_images folder inside the resnet50api project folder.

The returned result would be a JSON object telling us the classification of the image.

### Contact

Feel free to contact me (sadipgiri@bennington.edu) or open a ticket (PRs are always welcome!) with any questions, comments, suggestions, bug reports, etc.