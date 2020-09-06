import tensorflow as tf
from flask import Flask, make_response, request
server = Flask(__name__)


try:
    generator = tf.keras.models.load_model('inferencing')
except Exception as e:
    print('Ops..., something is wrong with your generator model.')
    print(e)


def make_img_response(img_bytes):
    response = make_response(img_bytes)
    response.mimetype = 'image/jpeg'

    return response


def decode(img_bytes):
    image = tf.image.decode_image(img_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image[20:-20, :]  # Crop operation, same as in training
    image = tf.image.resize(image, (128, 128))
    image = image * 2 - 1  # Normalization process

    return image


def encode(image):
    image = image * 0.5 + 0.5
    img_bytes = tf.image.encode_jpeg(image)

    return img_bytes


@server.route('/')
def mainpage():
    if request.method == 'GET':
        return '''
        <!doctype html>
            <title>Upload new File</title>
            <h1>Upload new File</h1>
            <form action="/process" method=post enctype=multipart/form-data id="form">
            <input type=file name=file>
            <input type=submit value=Upload>
            
            <label for="gender">Choose gender:</label>
            <select id="gender" name="genderoption" form="form">
                <option value="male">Male</option>
                <option value="female">Female</option>
            </select>
        </form>
        '''


@server.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        print(request.form)

        if 'file' not in request.files:
            return 'error'

        file = request.files['file']
        rawbuf = file.stream.read()

        image = decode(rawbuf)
        output_image = generator(tf.expand_dims(image, 0))[0]
        img_bytes = encode(output_image)

        response = make_img_response(img_bytes)

        return response
