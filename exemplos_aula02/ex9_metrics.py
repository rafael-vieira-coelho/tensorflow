from tensorflow.python.keras.models import Sequential 
from numpy import array 
from matplotlib import pyplot 
from tensorflow.python.keras.layers import Dense

X = array([ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
Y = array([ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
mod = Sequential()
mod.add(Dense(2, input_dim = 1))
mod.add(Dense(1, activation='sigmoid')) 
mod.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = 'acc')
# train the model 
history = mod.fit(X, Y, epochs = 400, batch_size = len(X), verbose = 2) 
# plot the metrics 
pyplot.plot(history.history['acc'], label='Accuracy')
pyplot.legend()
pyplot.show()



