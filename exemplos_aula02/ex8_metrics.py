from tensorflow.python.keras.models import Sequential 
from numpy import array 
from matplotlib import pyplot 
from tensorflow.python.keras.layers import Dense

# A sequence ​
X = array([ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# create a model 
mod = Sequential()
mod.add(Dense(2, input_dim = 1))
mod.add(Dense(1)) 
mod.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse', 'mae', 'mape', 'cosine_proximity'])
# train the model 
history = mod.fit(X, X, epochs = 500, batch_size = len(X), verbose = 2) 
# plot the metrics 
pyplot.plot(history.history['mse'], label='mean square error')
pyplot.plot(history.history['mae'], label='mean absolute error')
pyplot.plot(history.history['mape'], label='mean absolute percentage error')
pyplot.plot(history.history['cosine_proximity'], label='cosine proximity')
pyplot.legend()
pyplot.show()



