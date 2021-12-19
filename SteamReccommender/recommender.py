import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from fileHandling import *

try:
    PreviousLearned = tf.keras.models.load_model('ActionGoodIndieBad')
except:
    """ This would be learn model"""
    print("not learned model good bye")
    exit()

valuesfromFile(genericfiles())
