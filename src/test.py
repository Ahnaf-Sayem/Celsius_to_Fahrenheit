
import train
from train import model
import numpy as np
user_input=float(input('Enter your temperature in celsius:'))
print(model.predict(np.array([user_input])))