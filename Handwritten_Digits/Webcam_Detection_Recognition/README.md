What is this application?
=========================
It's a handwritten digits detector and recognizer that works in real time
on images captured from a webcam. 

How this application works?
===========================
The digit detection and recognition is based on two different deep neural networks.
The first network is an EAST (Efficient and Accurate Scene Text detection pipeline)
text detector meanwhile the second one is a  
convolutional neural network trained on the MNIST dataset.
EAST is capable of (1) running at near real-time at 13 FPS on 720p images and (2) 
obtains state-of-the-art text detection accuracy.

To run the application, you just need to connect a webcam to your computer 
and execute on your console:

```console
$ python3 real_time_digits.py
```
