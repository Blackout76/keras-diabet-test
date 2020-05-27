from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import keras
from datetime import datetime

# tensorboard --logdir ./logs

# load the dataset
dataset = loadtxt('pima-indians-diabetes.data.txt', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]


# define the keras model
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))# Layer 1 (input)
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(	loss='binary_crossentropy',
				optimizer='adam', 
				metrics=['accuracy'])

# TensorBoard clalback function visualization
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True, write_images=True)

# fit the keras model on the dataset
history = model.fit(X, y, epochs=500, batch_size=5, verbose=1, callbacks=[tbCallBack])
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))


# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(25):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))




# Save and load model
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# serialize model to JSON
modeldir = "models/model-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
model_json = model.to_json()
with open(modeldir, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# load json and create model
json_file = open(modeldir, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
for i in range(25):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))