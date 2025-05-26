import tensorflow as tf


model = tf.saved_model.load("parry_detector")
print(model.signatures)  # serving_default가 있는지 확인
