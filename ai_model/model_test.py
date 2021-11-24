from preprocessing import Preprocesser
import tensorflow as tf
import sys

p = Preprocesser()
model = tf.keras.models.load_model("./model/tf_model")
try:
    text = sys.argv[1]
except IndexError:
    text = input("User >> ")

output = model.generate(p.encoding(text))
print(p.decoding(output))
