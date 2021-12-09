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
    chatbot_data = pd.read_csv("data/raw_data/Chatbot/ChatbotData.csv", names=["dialogue", "response", "labels"]).loc[1:].drop(["labels"], axis=1)
    chatbot_data["dialogue"] = chatbot_data["dialogue"].apply(lambda row: row+"</s>")
    chatbot_data["response"] = chatbot_data["response"].apply(lambda row: "<s>"+row+"</s>")
    trainData = trainData.append(chatbot_data)
    print("ChatbotData Complete.")
    # emotional
    emotional_dataT = json.load(open("data/raw_data/emotional/emotional_Training.json", "r+", encoding="utf-8"))
    emotional_dataV = json.load(open("data/raw_data/emotional/emotional_Validation.json", "r+", encoding="utf-8"))
    for conv in emotional_dataT:
        temp = ""
        for key, sent in conv["talk"]["content"].items():
            if key[:2] == "SS":
                if sent.strip() == "" or len(temp.split("</s>")) > 5:
                    continue
                trainData = trainData.append(pd.DataFrame([[temp, "<s>"+sent+"</s>"]], columns=["dialogue", "response"]))
            temp += sent+"</s>"
    for conv in emotional_dataV:
        temp = ""
        for key, sent in conv["talk"]["content"].items():
            if key[:2] == "SS":
                if sent.strip() == "" or len(temp.split("</s>")) > 5:
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
                temp = "" if sentID == "1" or len(temp.split("</s>")) > 5 else temp
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
        self.lr = 3e-5
        self.batch_size = 16
        self.max_len = 170  # X - train_max = 150, val_max = 125 | y - train_max = ?, val_max = ?
        # data
        self.data_num = 103731  # train - 93478 + validation - 10253
        self.PREMODEL_NAME = "byeongal/Ko-DialoGPT"
        # tokenizers
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.PREMODEL_NAME)

    def getTrainData(self):
        # data's dialogue : S1</s>S2</s> | response : <s>R1</s>
        trainData = pd.read_csv("data/train.txt", sep="\t", names=["dialogue", "response"])
        trainData["response"] = trainData["response"].apply(lambda row: row[4:])

        train_x = self.tokenizer.batch_encode_plus(trainData["dialogue"].to_list(), return_tensors="tf",
                                                   max_length=self.max_len, padding="max_length", truncation=True)
        encoded_train_x = dict()
        for key, value in train_x.items():
            encoded_train_x[key] = value

        train_Y = self.tokenizer.batch_encode_plus((trainData["dialogue"] + trainData["response"]).to_list(), return_tensors="tf",
                                                   max_length=self.max_len, padding="max_length", truncation=True)["input_ids"]

        return tf.data.Dataset.from_tensor_slices((encoded_train_x, train_Y)).batch(self.batch_size).shuffle(256, seed=self.RANDOM_SEED)

    def getValidationData(self):
        validationData = pd.read_csv("data/validation.txt", sep="\t", names=["dialogue", "response"])
        validationData["response"] = validationData["response"].apply(lambda row: row[4:])

        validation_x = self.tokenizer.batch_encode_plus(validationData["dialogue"].to_list(), return_tensors="tf",
                                                        max_length=self.max_len, padding="max_length", truncation=True)
        encoded_validation_x = dict()
        for key, value in validation_x.items():
            encoded_validation_x[key] = value

        validation_Y = self.tokenizer.batch_encode_plus((validationData["dialogue"] + validationData["response"]).to_list(), return_tensors="tf",
                                                        max_length=self.max_len, padding="max_length", truncation=True)["input_ids"]

        return tf.data.Dataset.from_tensor_slices((encoded_validation_x, validation_Y)).batch(self.batch_size).shuffle(256, seed=self.RANDOM_SEED)

    def encoding(self, text: str):
        return self.tokenizer.encode(text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="tf")

    def decoding(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)
