import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
#load json data 
with open("intents.json") as file:
    data = json.load(file)
#if the model has already been trained then the moel will not the trained again and the preprocessing steps could be avoided
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    #different lists
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)#get all the words in the form of a list
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])
#Stemming a word is attempting to find the root of the word
#We will use this process of stemming words to reduce the vocabulary of our model and attempt to find the more general meaning behind sentences.
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

#tensorflow.reset_default_graph()
X_train = numpy.array(list(training[:, 0]))

y_train = numpy.array(list(training[:, 1]))
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")
with open('queries1.json',) as file:# test data file is loaded
   data1 = json.load(file)
words1 = []
doc_X = []
doc_y = []
classes1 = []
for query in data1["queries"]:
  tokens = nltk.word_tokenize(query["question"])
  words1.extend(tokens)
  doc_X.append(query["question"])
  doc_y.append(query["tag"])

  if query["tag"] not in labels:
        classes1.append(query["tag"])
words1 = [stemmer.stem(w.lower()) for w in words if w != "?"]
words1 = sorted(set(words1))
print("words1: ",words1)
print("doc_x: ",doc_X)
print("doc_y: ",doc_y)
print("classes1: ",classes1)
print(len(classes1))# 7 clases were there in the test dataset
training1 = []
out_empty = [0] * len(labels)
for idx, doc in enumerate(doc_X):
    bow = []
    text =[stemmer.stem(w.lower()) for w in doc]
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    # mark the index of class that the current pattern is associated
    # to
    output_row = list(out_empty)
    output_row[labels.index(doc_y[idx])] = 1
    # add the one hot encoded BoW and associated classes to training 
    training1.append([bow, output_row])
# shuffle the data and convert it to an array
print("output_row: ",output_row)
print("training1: ",training1)
training1 = numpy.array(training1, dtype=object)
# split the features and target labels
X_test = numpy.array(list(training1[:, 0]))
print("X_test",X_test)
print(X_test.shape)
y_test = numpy.array(list(training1[:, 1]))
print("y_test",y_test)
print(y_test.shape)
print(unique_labels(y_test))
#print(unique_labels(y_train))

y_pred=model.predict(X_test)
y_pred=numpy.argmax(y_pred, axis=1)
y_test=numpy.argmax(y_test, axis=1)
print("y_pred",y_pred)
print("y_test",y_test)
cm = confusion_matrix(y_true=y_test,y_pred=y_pred)
print(cm)
score=accuracy_score(y_test,y_pred)
print("accuracy=",score)
def plot(y_test,y_pred):
  labels=unique_labels(y_test)
 # column=[f'{label}' for label in labels]
 #indices=[f'{label}' for label in labels]
  print(labels[7])
  table=pd.DataFrame(confusion_matrix(y_test,y_pred),columns=labels,index=labels)
  true_p=0
  true_n=0
  false_p=0
  false_n=0
  arr=table.to_numpy()
  for i in range(len(arr)):
    for j in range(len(arr[i])):
      if (i==j) and (arr[i][j]!=0):
        true_p=true_p+arr[i][j]
      elif (i!=j) and (arr[i][j]==0):
        true_n=true_n+1
      else:
        false_p=false_p+1
        false_n=false_n+1

  false_n=false_n/2
  false_p=false_p/2

  print("true positive",true_p)
  print("true negative",true_n)
  print("false positive",false_p)
  print("false negative",false_n)
  return table

#print(plot(y_test,y_pred)) 


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type 'quit' to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index]>0.3:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print("Smart Bot: "+random.choice(responses))
        else:
            print("Smart Bot: I didn't get that, try again")

chat()