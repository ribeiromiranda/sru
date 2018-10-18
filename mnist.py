'''Trains a SRU on the MNIST dataset.

The parameters are not optimized.
'''
import keras
from keras.models import Model
from keras.layers import Dense, RNN, Input
from keras.datasets import mnist

from SRU import SRUCell

num_classes = 10

# input image dimensions
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


inputs = Input(x_train.shape[1:])
cell = SRUCell(num_stats=32, mavg_alphas=[0.0, 0.1, 0.3, 0.6, 0.9, 0.9999], recur_dims=8)
rnn = RNN([cell], return_sequences=False)(inputs)
output = Dense(num_classes, activation='softmax')(rnn)

model = Model(inputs=[inputs], outputs=[output])
model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=12, verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
