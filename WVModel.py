from gensim.models import word2vec

import logging
import os
import pickle


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data = list()

for files in os.listdir("./dataset/covid"):
    with open("./dataset/covid/" + files, "r") as f:
        x = f.readlines()
    data.extend(x)

for files in os.listdir("./dataset/non_covid"):
    with open("./dataset/non_covid/" + files, "r") as f:
        x = f.readlines()
    data.extend(x)

data.remove("\n")
print(len(data))

useless = {" ", "", "<p>", "<h>"}

dataset = []

for lines in data:
    start = lines.find(" <")
    lines = lines[start:].lower()
    words = lines.split(" ")
    x = []
    for word in words:
        if (word in useless) or (word < 'a' or word > 'z'):
            continue
        else:
            if "-" in word:
                word = word.replace("-", "")
            word = word.strip(",")
            if "19" in word:
                word = word.replace("19", "")
            x.append(word.strip("."))
    dataset.append(x)

print("start training")

model = word2vec.Word2Vec(dataset, min_count = 10, size = 200)

print("end traning")

model.save("./W2V.model")

print(model['coronavirus'])