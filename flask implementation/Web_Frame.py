from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from keras_resnet.models import ResNet50
from keras.optimizers import adam_v2
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

vocab = np.load('vocab.npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}


print("+"*50)
print("vocabulary loaded")


embedding_size = 128
vocab_size = len(vocab)
max_len = 40


image_model = Sequential()

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))


language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))


conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.load_weights('mine_model_weights.h5')

print("="*150)
print("MODEL LOADED")

#gharjao = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')


gharjao = load_model('model.h5')

print("="*150)
print("RESNET MODEL LOADED")

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        print('display_image filename: ' + filename)
        global model, gharjao, vocab, inv_vocab
        image = cv2.imread('static/uploads/1.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.reshape(image, (1, 224, 224, 3))
        
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


'''@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    global model, gharjao, vocab, inv_vocab

    image = cv2.imread('static/uploads/1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.reshape(image, (1, 224, 224, 3))
    incept = gharjao.predict(image).reshape(1, 2048)
    print("=" * 50)
    print("Predict Features")
    text_in = ['startofseq']
    final = ''
    print("=" * 50)
    print("GETING Captions")
    count = 0
    while tqdm(count < 20):
        count += 1
        encoded = []
        for i in text_in:
            encoded.append(vocab[i])
        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1, max_len)
        sampled_index = np.argmax(model.predict([incept, padded]))
        sampled_word = inv_vocab[sampled_index]
        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word
        text_in.append(sampled_word)

    return redirect(url_for('static', filename='uploads/' + filename, final=final), code=301 )
'''

if __name__ == "__main__":
    app.run()