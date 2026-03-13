
import train
import numpy
from train import model
user_input=float(input('Enter your temperature in celsius:'))
print(model.predict(numpy.array([user_input])))