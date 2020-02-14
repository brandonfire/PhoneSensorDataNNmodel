import sys
import numpy as np
import tensorflow as tf
#from sklearn.linear_model import LinearRegression



def readinputfile(TrainX,TrainY): #read trainX and trainY
    in_filex = open(TrainX,'r')
    in_filey = open(TrainY,'r')
    X = list(np.loadtxt(TrainX,dtype=np.single))
    for i in range(len(X)):
        X[i] = np.pad(X[i], (7, 8), 'constant', constant_values=(0, 0))
    Y = np.loadtxt(TrainY)
    Y = Y - 1
    in_filex.close()
    in_filey.close()
    return np.asarray(X),Y


def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(24,24,1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(12)
  ])
  
def main():
    X,Y = readinputfile('X_train.txt','y_train.txt')
    X = X.reshape(len(Y),24,24,1)
    X = tf.keras.utils.normalize(X)
    testX, testY = readinputfile('X_test.txt','y_test.txt')
    testX = testX.reshape(len(testY),24,24,1)
    testX = tf.keras.utils.normalize(testX)
    model = create_model()
    #print(X[0])
    model.compile(
              optimizer=tf.keras.optimizers.Adam(0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    log_dir="./logs/fit"
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(x=X, 
          y=Y, 
          epochs=1000,
          batch_size = 100,
          validation_data=(testX, testY), 
          callbacks=[tensorboard_callback])
    model.evaluate(testX, testY)
    
    
    
    
if __name__ == "__main__":
    main()    
