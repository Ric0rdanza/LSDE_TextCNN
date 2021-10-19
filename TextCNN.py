from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from gensim.models import word2vec

import os
import numpy as np
import tensorflow as tf

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

useless = {" ", "", "<p>", "<h>"}

wv = word2vec.Word2Vec.load("./W2V.model")

def preprocess(sentences):
    if "\n" in sentences:
        sentences.remove("\n")
    dataset = []
    for sentence in sentences:
        start = sentence.find(" <")
        sentence = sentence[start:].lower()
        words = sentence.split(" ")
        wordvec = []
        for word in words:
            if len(wordvec) == 200:
                break
            else:
                if (word in useless) or (word < 'a' or word > 'z'):
                    continue
                else:
                    word = word.strip(",")
                    word = word.strip(".")
                    if "-" in word:
                        word = word.replace("-", "")
                    if "19" in word:
                        word = word.replace("19", "")
                    try:
                        x = wv[word]
                    except:
                        continue
                wordvec.append(x)
        if len(wordvec) < 200:
            wordvec.extend([[0 for _ in range(200)] for _ in range(200 - len(wordvec))])
        dataset.append(wordvec)
    return dataset

def TextCNN():
    InputLayer = Input(shape = [200, 200, 1])
    Conv50 = Conv2D(filters = 32, kernel_size = (50, 200), strides = 1, padding = 'valid', activation = "relu")(InputLayer)
    MaxP50 = MaxPooling2D(pool_size = (151, 1))(Conv50)
    F1 = Flatten()(MaxP50)

    Conv35 = Conv2D(filters = 32, kernel_size = (35, 200), strides = 1, padding = 'valid', activation = "relu")(InputLayer)
    MaxP35 = MaxPooling2D(pool_size = (166, 1))(Conv35)
    F2 = Flatten()(MaxP35)
    
    Conv20 = Conv2D(filters = 32, kernel_size = (20, 200), strides = 1, padding = 'same', activation = "relu")(InputLayer)
    MaxP20 = MaxPooling2D(pool_size = (181, 1))(Conv20)
    F3 = Flatten()(MaxP20)

    Conv15 = Conv2D(filters = 32, kernel_size = (15, 200), strides = 1, padding = 'same', activation = "relu")(InputLayer)
    MaxP15 = MaxPooling2D(pool_size = (186, 1))(Conv15)
    F4 = Flatten()(MaxP15)

    Conv7 = Conv2D(filters = 32, kernel_size = (7, 200), strides = 1, padding = 'same', activation = "relu")(InputLayer)
    MaxP7 = MaxPooling2D(pool_size = (194, 1))(Conv7)
    F5 = Flatten()(MaxP7)

    Conv3 = Conv2D(filters = 32, kernel_size = (3, 200), strides = 1, padding = 'same', activation = "relu")(InputLayer)
    MaxP3 = MaxPooling2D(pool_size = (197, 1))(Conv3)
    F6 = Flatten()(MaxP3)

    D = concatenate([F1, F2, F3, F4, F5, F6])

    out = Dense(2, activation = "sigmoid")(D)

    model = Model(InputLayer, out)
    return model

def EarlyStopping():
    return tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy", min_delta = 0.03, patience = 3, restore_best_weights = True)

def CheckPoint():
    return tf.keras.callbacks.ModelCheckpoint("./model.h5", monitor="accuracy", save_best_only= True, save_freq = 1)

def ClassifierTest(test):
    # test => The text need to be classified
    test = "@@85043941 <p> Hall chairman Jerry Colangelo says the ceremony for the 2020 class will be moved to \
            next year because of the coronavirus . <p> SPRINGFIELD , Mass. -- Naismith Memorial Basketball Hall\
            of Fame chairman Jerry Colangelo told ESPN that the enshrinement ceremony for Kobe Bryant , Kevin G\
            arnett , Tim Duncan and five others in this year 's class will be delayed until 2021 because of the\
           coronavirus pandemic . <p> The ceremony was to have taken place Aug. 29 in Springfield . Colangelo t\
           old ESPN the event will be moved to 2021 . <p> There was no announcement from the Hall of Fame about\
            rescheduling . <p> \" We 're definitely canceling , \" Colangelo told ESPN . \" It 's going to have to\
           be the first quarter of next year . We 'll meet in a couple of weeks and look at the options of how \
           and when and where . \" <p> Bryant , who died in a January helicopter crash , got into the Hall in hi\
           s first year as finalist , as did Garnett , Duncan and WNBA great Tamika Catchings . The other membe\
           rs of the class of 2020 are two-time @ @ @ @ @ @ @ @ @ @ coach Kim Mulkey , 1,000-game winner Barbar\
           a Stevens of Bentley and three-time Final Four coach Eddie Sutton , who died last weekend . "

    x = [test]

    test_x = preprocess(x)

    classifyer = TextCNN()
    classifyer.load_weights("./model.h5")

    print(classifyer.predict(np.array(test_x)))

def Training():
    covid = []
    non_covid = []

    val_covid = []
    val_non_covid = []

    for files in os.listdir("./dataset/covid"):
        with open("./dataset/covid/" + files, "r") as f:
            covid.extend(f.readlines())
            covid = covid[0: 4501]

    covid_y = [1 for _ in range(len(covid) - 1)]

    for files in os.listdir("./dataset/non_covid"):
        with open("./dataset/non_covid/" + files, "r") as f:
            non_covid.extend(f.readlines())
            non_covid = non_covid[0: 4501]
    non_covid_y = [0 for _ in range(len(non_covid) - 1)]

    covid_x = preprocess(covid)
    non_covid_x = preprocess(non_covid)

    val_covid_x = covid_x[4000: 4500]
    val_non_covid_x = non_covid_x[4000: 4500]

    val_covid_y = [1 for _ in range(len(val_covid_x))]
    val_non_covid_y = [0 for _ in range(len(val_non_covid_x))]

    val_covid_x.extend(val_non_covid_x)
    val_covid_y.extend(val_non_covid_y)


    covid_x.extend(non_covid_x)
    covid_y.extend(non_covid_y)

    train_x = np.array(covid_x)
    train_y = np.array(covid_y)

    val_x = np.array(val_covid_x)
    val_y = np.array(val_covid_y)


    model = TextCNN()
    model.summary()
    model.compile(optimizer = "sgd", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    model.fit(train_x, train_y, 32, epochs = 10, callbacks = [EarlyStopping(), CheckPoint()], validation_data=(val_x, val_y))


if __name__ == "__main__":
    ClassifierTest("classify test")