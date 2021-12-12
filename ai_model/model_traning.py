from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from transformers import TFGPT2LMHeadModel
from preprocessing import Preprocesser
import tensorflow as tf

def lr_scheduler(epoch, lr):
    if epoch < 2:
        return 5e-5
    elif epoch < 4:
        return 3e-5
    else:
        return 1e-5

class DialoGPT(tf.keras.Model, TFGPT2LMHeadModel):
    def __init__(self, *args, **kwargs):
        super(DialoGPT, self).__init__(*args, **kwargs)
        self.koDialoGPT = TFGPT2LMHeadModel.from_pretrained(p.PREMODEL_NAME, from_pt=True)

        self.max_len = p.max_len
        self.batch_size = p.batch_size

    def call(self, inputs, training=None, mask=None):
        # {'input_ids': (batch, max_len), 'attention_mask': (batch, max_len)
        # -> (batch, max_len, vocab_size(51200))
        output = self.koDialoGPT(inputs, return_dict=True)
        return output.logits

    def train_step(self, data):
        pass

    def test_step(self, data):
        pass

    def get_config(self):
        return self.koDialoGPT.config


if __name__ == "__main__":
    p = Preprocesser()
    # model = TFGPT2LMHeadModel.from_pretrained(p.PREMODEL_NAME, from_pt=True)
    model = DialoGPT()
    history = ""
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    for optim in ["adam", "rmsprop", "nadam"]:
        model.compile(loss=loss, optimizer=optim, metrics="accuracy")
        hist = model.fit(p.getTrainData(), validation_data=p.getValidationData(), batch_size=p.batch_size, epochs=5,
                         callbacks=[EarlyStopping(patience=3), LearningRateScheduler(lr_scheduler),
                                    ModelCheckpoint("./model/"+optim+"_model", monitor="val_accuracy", save_best_only=True)])
        model.save("./model/"+optim+"_last_model.h5")
        history += optim + " : " + str(hist) + "\n\n"
    open("./data/history.txt", "w+", encoding="utf-8").write(history)
