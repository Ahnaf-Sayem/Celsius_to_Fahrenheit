import learn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

c_train= np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
f_train= np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)
c_val=np.array([18, 19, 39, 59, 69, 129],dtype=float)
f_val=np.array([64.4, 66.2, 102.2, 138.2, 156.2, 264.2],dtype=float)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1), metrics=['mean_squared_error'])

history = model.fit(c_train,f_train, epochs=500,validation_data=(c_val,f_val), verbose=False)
print("Finished training the model")
fig,ax = plt.subplots(2,1,figsize=(8,12))
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()