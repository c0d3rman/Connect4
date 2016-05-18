import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

trainingDataPercentage = 0.8
dropoutRate = 0
dropoutRateInput = 0
hiddenUnits = 410
hiddenLayers = 3
batchSize = 64
epochs = 20
patience = 2
opt = 'adam'
activation = 'relu'
initialization = 'uniform'
loss = 'categorical_crossentropy'

# Get raw data from file
dataset = np.array(np.genfromtxt("connect-4.data", delimiter=',', dtype=None))

# Shuffle the data and split into input/output
np.random.shuffle(dataset)
X_all = dataset[:,:-1]
Y_all = dataset[:,-1]

# Convert the to the appropriate format
def Xmap(item):
	if item == "b":
		return 0
	elif item == "x":
		return 1
	elif item == "o":
		return -1
	else:
		print "Uh oh! Unexpected value in X"
vXmap = np.vectorize(Xmap)
X_all = vXmap(X_all)

temp = []
for i in xrange(Y_all.shape[0]):
	if Y_all[i] == "win":
		temp.append([1, 0, 0])
	elif Y_all[i] == "loss":
		temp.append([0, 1, 0])
	elif Y_all[i] == "draw":
		temp.append([0, 0, 1])
	else:
		print "Uh oh! Unexpected value in Y"
Y_all = np.array(temp)

# Set the type of the data
X_all = X_all.astype('int32')
Y_all = Y_all.astype('int32')

# Split the data into testing and training
XSplitIndex = int(trainingDataPercentage * X_all.shape[0])
YSplitIndex = int(trainingDataPercentage * Y_all.shape[0])
X_train, X_test = X_all[:XSplitIndex,:], X_all[XSplitIndex:,:]
Y_train, Y_test = Y_all[:YSplitIndex,:], Y_all[YSplitIndex:,:]

# Print number of samples
print X_train.shape[0], 'train samples'
print X_test.shape[0], 'test samples'


# Construct model
model = Sequential()
model.add(Dense(hiddenUnits, input_shape=(X_train.shape[1],), init=initialization))
model.add(Activation(activation))
model.add(Dropout(dropoutRateInput))
for i in xrange(hiddenLayers):
	model.add(Dense(hiddenUnits, init=initialization))
	model.add(Activation(activation))
	model.add(Dropout(dropoutRate))
model.add(Dense(Y_train.shape[1], init=initialization))
model.add(Activation('softmax'))

#model.summary()

# Compile and train
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=batchSize, nb_epoch=epochs, verbose=1, validation_data=(X_test, Y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=patience)])

# Report results
score = model.evaluate(X_test, Y_test, verbose=0)
print 'Test score:', score[0]
print 'Test accuracy:', score[1]
