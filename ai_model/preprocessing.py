from transformers import GPT2TokenizerFast
import tensorflow as tf
import pandas as pd
import json
import os

def make_dataset():
    file_name = ""

    raw_data = json.load(open('./data/raw_data/'+file_name+".json", "r+", encoding="utf-8"))
    # raw_data = open('./data/raw_data/'+file_name+".txt", "r+", encoding="utf-8").read()


class Preprocesser:
    def __init__(self):
        self.RANDOM_SEED = 10
        # HyperParam
        self.lr = 1e-5
        self.batch_size = 16
        self.embedding_dim = 128
        self.input_dim = 0
        self.output_dim = 0
        # data
        self.data_num = 0
        self.PREMODEL_NAME = "byeongal/Ko-DialoGPT"
        # dialogue : S1</s>S2</s> | response : <s>R1</s>
        self.trainData = pd.read_csv("./data/train.txt", sep="\t", names=["dialogue", "response"])
        self.validationData = pd.read_csv("./data/validation.txt", sep="\t", names=["dialogue", "response"])
        self.testData = pd.read_csv("./data/test.txt", sep="\t", names=["dialogue", "response"])
        # tokenizers
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.PREMODEL_NAME)

    def getTrainData(self):
        train_x = self.tokenizer.batch_encode_plus(self.trainData["dialogue"], return_tensors="tf",
                                                   max_length=self.input_dim, padding="max_length", truncation=True)
        train_y = self.tokenizer.batch_encode_plus(self.trainData["response"], return_tensors="tf",
                                                   max_length=self.output_dim, padding="max_length", truncation=True)

        return tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(self.batch_size).shuffle(1000, seed=self.RANDOM_SEED)

    def getValidationData(self):
        validation_x = self.tokenizer.batch_encode_plus(self.validationData["dialogue"], return_tensors="tf",
                                                        max_length=self.input_dim, padding="max_length", truncation=True)
        validation_y = self.tokenizer.batch_encode_plus(self.validationData["response"], return_tensors="tf",
                                                        max_length=self.output_dim, padding="max_length", truncation=True)

        return tf.data.Dataset.from_tensor_slices((validation_x, validation_y)).batch(self.batch_size).shuffle(1000, seed=self.RANDOM_SEED)

    def getTestData(self):
        test_x = self.tokenizer.batch_encode_plus(self.testData["dialogue"], return_tensors="tf",
                                                  max_length=self.input_dim, padding="max_length", truncation=True)
        test_y = self.tokenizer.batch_encode_plus(self.testData["response"], return_tensors="tf",
                                                  max_length=self.output_dim, padding="max_length", truncation=True)

        return tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(self.batch_size).shuffle(1000, seed=self.RANDOM_SEED)

    def encoding(self, text: str):
        return self.tokenizer.encode(text + self.tokenizer.eos_token)

    def decoding(self, ids: list[int]):
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)
