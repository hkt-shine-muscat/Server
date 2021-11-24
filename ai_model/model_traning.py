from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from preprocessing import Preprocesser
from transformers import TFAutoModel
import tensorflow as tf

p = Preprocesser()
model = TFAutoModel.from_pretrained(p.PREMODEL_NAME)

optim = tf.keras.optimizers.Adam(learning_rate=p.lr)
loss = tf.keras.losses.sparse_categorical_crossentropy(from_logits=True)
model.compile(loss=loss, optimizer=optim, metrics="accuracy")
model.fit(p.getTrainData(), validation_data=p.getValidationData(), batch_size=p.batch_size, epochs=4,
          callbacks=[EarlyStopping(patience=3), ModelCheckpoint("./model/tf_model", monitor="val_accuracy", save_best_only=True)])
model.save("./model/last_model.h5")
