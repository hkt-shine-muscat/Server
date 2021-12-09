from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from transformers import TFAutoModelForCausalLM
from preprocessing import Preprocesser
import tensorflow as tf

def lr_scheduler(epoch, lr):
    if epoch < 2:
        return 5e-5
    elif epoch < 4:
        return 3e-5
    else:
        return 1e-5


if __name__ == "__main__":
    p = Preprocesser()
    model = TFAutoModelForCausalLM.from_pretrained(p.PREMODEL_NAME, from_pt=True)
    history = ""
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    for optim in ["adam", "rmsprop", "nadam"]:
        model.compile(loss=loss, optimizer=optim, metrics="accuracy")
        hist = model.fit(p.getTrainData(), validation_data=p.getValidationData(), batch_size=p.batch_size, epochs=5,
                         callbacks=[EarlyStopping(patience=3), LearningRateScheduler(lr_scheduler),
                                    ModelCheckpoint("./model/"+optim+"_model", monitor="val_accuracy", save_best_only=True)])
        model.save("./model/"+optim+"_last_model.h5")
        history += optim + " : " + str(hist) + "\n\n"
    open("./data/history.txt", "w+", encoding="utf-8").write(history)
