## 错误：

Could not interpret optimizer identifier: <tensorflow.python.keras.optimizer_v2.adam.Adam object at 0x7f7386b3f850>

原因： import tensorflow as tf时，使用optimizer= tf.keras.optimizers.Adam(), model.compile(optimizer=optimizer)会无法解析， 需要改成 from
tensorflow import keras, optimizer = keras.optimizers.Adam()