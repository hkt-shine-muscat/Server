from transformers import GPT2TokenizerFast
import tensorflow as tf
import pandas as pd
import random
import json
import os

def make_dataset():
    trainData = pd.DataFrame(columns=["dialogue", "response"])
    valData = pd.DataFrame(columns=["dialogue", "response"])
    print("Data Making Start.")
    # ChatbotData
    chatbot_data = pd.read_csv("data/raw_data/ChatbotData.csv", names=["dialogue", "response", "labels"]).loc[1:].drop(["labels"], axis=1)
    chatbot_data["dialogue"] = chatbot_data["dialogue"].apply(lambda row: row+"</s>")
    chatbot_data["response"] = chatbot_data["response"].apply(lambda row: "<s>"+row+"</s>")
    trainData = trainData.append(chatbot_data)
    print("ChatbotData Complete.")
    # emotional
    emotional_dataT = json.load(open("./data/raw_data/emotional_Training.json", "r+", encoding="utf-8"))
    emotional_dataV = json.load(open("./data/raw_data/emotional_Validation.json", "r+", encoding="utf-8"))
    for conv in emotional_dataT:
        temp = ""
        for key, sent in conv["talk"]["content"].items():
            if key[:2] == "SS":
                if sent.strip() == "" or len(sent.split("</s>")) > 5 :
                    continue
                trainData = trainData.append(pd.DataFrame([[temp, "<s>"+sent+"</s>"]], columns=["dialogue", "response"]))
            temp += sent+"</s>"
    for conv in emotional_dataV:
        temp = ""
        for key, sent in conv["talk"]["content"].items():
            if key[:2] == "SS":
                if sent.strip() == "" or len(sent.split("</s>")) > 5 :
                    continue
                valData = valData.append(pd.DataFrame([[temp, "<s>"+sent+"</s>"]], columns=["dialogue", "response"]))
            temp += sent+"</s>"
    print("EmotionalData Complete.")
    # koreanConversation
    for filename in os.listdir("./data/raw_data/koreanConversation"):
        temp = ""
        ks_data = pd.read_excel("./data/raw_data/koreanConversation/"+filename)[["SENTENCE", "SENTENCEID", "QA"]]
        try:
            for sent, sentID, QA in ks_data.loc:
                temp = "" if sentID == "1" or len(sent.split("</s>")) > 5 else temp
                if QA == "A":
                    if sent.strip() == "":
                        continue
                    if random.randint(1, 10) > 4:
                        trainData = trainData.append(pd.DataFrame([[temp, "<s>" + sent + "</s>"]], columns=["dialogue", "response"]))
                    else:
                        valData = valData.append(pd.DataFrame([[temp, "<s>" + sent + "</s>"]], columns=["dialogue", "response"]))
                temp += sent + "</s>"
        except KeyError:
            continue
    print("KoreanConversation Complete.")
    # # kcs
    # trainData = pd.DataFrame(columns=["dialogue", "response"])
    # valData = pd.DataFrame(columns=["dialogue", "response"])
    # for TorV in ["train", "valid"]:
    #     print("KCS " + TorV + " start.")
    #     for filename in os.listdir("./data/raw_data/kcs_"+TorV):
    #         kcs_data = json.load(open("./data/raw_data/kcs_"+TorV+"/"+filename, "r+", encoding="utf-8"))["data"]
    #         print(TorV+" "+filename+" start.")
    #         for conv in kcs_data:
    #             temp = ""
    #             temp_sent = ""
    #             pre_turn = ""
    #             for dialogue in conv["body"]["dialogue"]:
    #                 if dialogue["turnID"] != pre_turn:
    #                     if dialogue["turnID"] == "T1":
    #                         if TorV == "train":
    #                             trainData = trainData.append(pd.DataFrame([[temp, "<s>" + temp_sent + "</s>"]], columns=["dialogue", "response"]))
    #                         else:
    #                             valData = valData.append(pd.DataFrame([[temp, "<s>" + temp_sent + "</s>"]], columns=["dialogue", "response"]))
    #                         temp = ""
    #                         temp_sent = ""
    #                     elif int(dialogue["turnID"][1:]) % 2 == 0:
    #                         temp += temp_sent + "</s>"
    #                     else:
    #                         if TorV == "train":
    #                             trainData = trainData.append(pd.DataFrame([[temp, "<s>" + temp_sent + "</s>"]], columns=["dialogue", "response"]))
    #                         else:
    #                             valData = valData.append(pd.DataFrame([[temp, "<s>" + temp_sent + "</s>"]], columns=["dialogue", "response"]))
    #                         temp += temp_sent + "</s>"
    #                 temp_sent += dialogue["utterance"] + " "
    # print("KoreanConversationSummary Complete.")
    # save
    trainData.to_csv("./data/train.txt", sep="\t")
    valData.to_csv("./data/validation.txt", sep="\t")
    print("All Save Complete.")


class Preprocesser:
    def __init__(self):
        self.RANDOM_SEED = 10
        # HyperParam
        self.lr = 1e-5
        self.batch_size = 16
        self.embedding_dim = 256
        self.input_dim = None
        self.output_dim = None
        # data  # dialogue : S1</s>S2</s> | response : <s>R1</s>  #  preprocessing input's data?
        self.data_num = None
        self.PREMODEL_NAME = "byeongal/Ko-DialoGPT"
        self.trainData = pd.read_csv("./data/train.txt", sep="\t", names=["dialogue", "response"])
        self.validationData = pd.read_csv("./data/validation.txt", sep="\t", names=["dialogue", "response"])
        # tokenizers
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.PREMODEL_NAME)

    def getTrainData(self):
        train_x = self.tokenizer.batch_encode_plus(self.trainData["dialogue"], return_tensors="tf", add_special_tokens=False,
                                                   max_length=self.input_dim, padding="max_length", truncation=True)
        train_y = self.tokenizer.batch_encode_plus(self.trainData["response"], return_tensors="tf", add_special_tokens=False,
                                                   max_length=self.output_dim, padding="max_length", truncation=True)

        return tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(self.batch_size).shuffle(1000, seed=self.RANDOM_SEED)

    def getValidationData(self):
        validation_x = self.tokenizer.batch_encode_plus(self.validationData["dialogue"], return_tensors="tf", add_special_tokens=False,
                                                        max_length=self.input_dim, padding="max_length", truncation=True)
        validation_y = self.tokenizer.batch_encode_plus(self.validationData["response"], return_tensors="tf", add_special_tokens=False,
                                                        max_length=self.output_dim, padding="max_length", truncation=True)

        return tf.data.Dataset.from_tensor_slices((validation_x, validation_y)).batch(self.batch_size).shuffle(1000, seed=self.RANDOM_SEED)

    def encoding(self, text: str):
        return self.tokenizer.encode(text + self.tokenizer.eos_token)

    def decoding(self, ids: list[int]):
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)




