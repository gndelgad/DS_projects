What is this application?
=========================
It's a web service deployed on GCP App Engine that
recognizes a handwritten digit from an uploaded image.  

How this application works?
===========================
The digit recognition is based on a deep neural network trained on the MNIST dataset.
The user has two models of DNN to chose: 1) A multi-layered TensorFlow 
perceptron with 2 hidden layers 
or a convolutional neural network trained and deployed with Keras.

What are the limitations of this application?
=============================================
There must be only one digit in the uploaded image.


How to deploy this web service in GCP App Engine?
=================================================
Follow the below steps:

*	Open your GCP console, access to "my_project" and activate the cloud shell.
*	Copy the files into your project's default compute engine instance.
*	Run in the console:
	```console
	$ gcloud app deploy app.yaml --project my_project
	```