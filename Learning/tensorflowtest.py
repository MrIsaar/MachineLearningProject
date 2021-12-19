import pandas as pd
import numpy as np
from proccessFiles import genericfiles

# Make numpy values easier to read.
#np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
#print("TensorFlow version:", tf.__version__)
from tensorflow.keras import layers

trainingFile = genericfiles("steam","trainAction.csv")

trainSamples = pd.read_csv(
    trainingFile,encoding="utf-8",delimiter=',',quotechar='"',skipinitialspace=True,
    names=["appid","name","release_date","english","developer","publisher","platforms","required_age","categories","genres","steamspy_tags","achievements","positive_ratings","negative_ratings","average_playtime","median_playtime","owners","price","label"]
    )
train_features = trainSamples.copy()
train_labels = train_features.pop('label')

# Create a symbolic input
input = tf.keras.Input(shape=(), dtype=tf.float32)

# Perform a calculation using the input
result = 2*input + 1

# the result doesn't have a value
#print(result)

calc = tf.keras.Model(inputs=input, outputs=result)

#print(calc(1).numpy())
#print(calc(2).numpy())

inputs = {}

for name, column in train_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

#print(inputs)

numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(trainSamples[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

#print(all_numeric_inputs)

preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue

    lookup = layers.StringLookup(vocabulary=np.unique(train_features[name]))
    one_hot = layers.CategoryEncoding(max_tokens=lookup.vocab_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)


preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

train_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

#tf.keras.utils.plot_model(model = train_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

train_features_dict = {name: np.array(value) for name, value in train_features.items()}

features_dict = {name:values[:1] for name, values in train_features_dict.items()}
train_preprocessing(features_dict)

def train_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
  ])

    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.optimizers.Adam())
    return model

train_model = train_model(train_preprocessing, inputs)

train_model.fit(x=train_features_dict, y=train_labels, epochs=10)

train_model.save('ActionGoodIndieBad')
reloaded = tf.keras.models.load_model('ActionGoodIndieBad')

index = 480
size = 20
features_dict = {name:values[index:size+index] for name, values in train_features_dict.items()}
before = train_model(features_dict)
after = reloaded(features_dict)
print("b: ",before)
print("a: ",after)
assert (before[0]-after[0])<1e-3

print("done")