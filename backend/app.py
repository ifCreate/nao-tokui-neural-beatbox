import numpy as np
import json
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization,Activation
from keras.layers.advanced_activations import ELU
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import load_model
from flask import Flask
from flask_cors import *
import multiprocessing

app = Flask(__name__)
CORS(app, supports_credentials=True)

data = np.load('./data/drum_data_128.npz')
NB_CLASS = 9
drum_melspecs = data['melspecs']
drum_genres = data['genres']
class_weight = {}
total = drum_genres.shape[0]
for i in range(NB_CLASS):
    nb = np.sum(np.argmax(drum_genres, axis=1) == i)
    class_weight[i] = total / float(nb)
nb_total = drum_melspecs.shape[0]
nb_train = int(nb_total * 0.9)
train_melspecs = drum_melspecs[:nb_train]
train_genres = drum_genres[:nb_train]
val_melspecs = drum_melspecs[nb_train:]
val_genres = drum_genres[nb_train:]


q = multiprocessing.Queue()

def train_worker():
    es = EarlyStopping(verbose=1, patience=5)
    model = load_model('./model/drum_spec_model.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])
    hist = model.fit(train_melspecs, train_genres, batch_size=4,
                 epochs=10, verbose=1, shuffle=False,
                 validation_data = (val_melspecs, val_genres), class_weight=class_weight, callbacks=[es])
    q.put(json.dumps(hist.history))


@app.route('/')
def hello_world():
   return 'Hello World'


@app.route('/train')
def train():
    p = multiprocessing.Process(target=train_worker)
    p.start()
    return q.get()


if __name__ == '__main__':
   app.run(debug=False)
