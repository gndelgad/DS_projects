from flask import Flask, render_template, request
import skimage as ski
from cv2 import cvtColor, boundingRect, COLOR_BGR2GRAY
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.models import load_model
from tensorflow.compat.v1.keras import backend
tf.disable_v2_behavior()

USE_CNN = True

if USE_CNN:
    session = tf.Session(graph=tf.Graph())
    with session.graph.as_default():
        # with context manager operations are added to the session graph.
        # The need of using context managers here and for prediction is because
        # the way Keras works with different threads
        backend.set_session(session)
        cnn_model = load_model("./model/cnn_model.h5")

else:
    # Tensorflow graph init
    saver = tf.train.import_meta_graph("model/my_model_final.ckpt.meta")
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    logits = graph.get_tensor_by_name("dnn/outputs/BiasAdd:0")
    softmax = graph.get_tensor_by_name("dnn/Softmax:0")

app = Flask(__name__, template_folder='templates')


@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/uploader', methods=['POST'])
def upload_image_file():
    if request.method == 'POST':
        im2arr = fromPathMnistTrasformation(request.files['file'].stream)

        if USE_CNN:
            number, prob_number = cnnPrediction(im2arr)
        else:
            number, prob_number = dnnPrediction(im2arr)

        return 'Predicted Number: ' + number + ', with prob: ' + prob_number


def padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector


def mnistTransformation(image, angle_rotation):
    if len(image.shape) == 3:
        image = cvtColor(image, COLOR_BGR2GRAY)
    thresh = ski.filters.threshold_otsu(image)
    image = 255 * np.where(image < thresh, (1 - image), 0).astype(np.uint8)
    (x, y, w, h) = boundingRect(image)
    image = image[y: y + h, x: x + w]
    image = ski.util.pad(image, int(0.35 * max(image.shape)), padwithtens)
    if angle_rotation:
        image = ski.transform.rotate(image, angle_rotation)
    image = ski.transform.resize(image, (28, 28), anti_aliasing=True)
    image = ski.restoration.denoise_tv_chambolle(image, 1e-2)
    image = 255 * image / image.max()
    return image


def fromPathMnistTrasformation(im_path, angle_rotation=None):
    image = ski.io.imread(im_path)
    return mnistTransformation(image, angle_rotation)


def cnnPrediction(im2arr):
    with session.graph.as_default():
        backend.set_session(session)
        pred = cnn_model.predict(im2arr.reshape(1, 28, 28, 1) / 255)

    return str(pred.argmax()), str(pred.max())


def dnnPrediction(im2arr):
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, "model/my_model_final.ckpt")
        im2arr_scaled = im2arr.reshape(-1, 28 * 28) / 255.0
        Z = logits.eval(feed_dict={X: im2arr_scaled})
        probs = softmax.eval(feed_dict={X: im2arr_scaled})

        y_pred = [[row[-1], row[-2]] for row in np.argsort(Z, axis=1)]
        prob_pred = [[row[-1], row[-2]] for row in np.sort(probs, axis=1)]

    return str(y_pred[0]), str(prob_pred[0])


if __name__ == '__main__':
    print(("* Loading Tensorflow model and Flask starting server..."
           "please wait until server has fully started"))

    app.run(host='127.0.0.1', port=8080, debug=True)
