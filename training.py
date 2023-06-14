import random
import json
import pickle
import numpy as np
import nltk

from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

lematizador = WordNetLemmatizer()

# Arquivo de pretenções
intents = json.loads(open("intents.json").read())

#Armazenamento de dados
palavras = []
classes = []
docs = []
ignorar_caracteres = ["?", "!", ".", ","]
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #Separando as palavras dos padrões
        lista_palavras = nltk.word_tokenize(pattern)
        palavras.extend(lista_palavras)  #Criando uma lista de palavras

        #Associar padrões com seu rótulo
        docs.append(((lista_palavras), intent['tag']))

        #Adicionando rótulos à lista de classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#Armazenando o lema, ou radical, de cada palavra
palavras = [lematizador.lemmatize(palavra)
         for palavra in palavras if palavra not in ignorar_caracteres]
palavras = sorted(set(palavras))

#Salvando a lista de palavras e classes em um arquivo binário para trabalhar com valores numéricos das palavras
pickle.dump(palavras, open('palavras.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
treinamento = []
output_vazio = [0] * len(classes)
for doc in docs:
    bag = []
    padrao_palavras = doc[0]
    padrao_palavras = [lematizador.lemmatize(
        palavra.lower()) for palavra in padrao_palavras]
    for palavra in palavras:
        bag.append(1) if palavra in padrao_palavras else bag.append(0)

    #Fazendo uma cópia do output_vazio
    output_row = list(output_vazio)
    output_row[classes.index(doc[1])] = 1
    treinamento.append([bag, output_row])
random.shuffle(treinamento)
treinamento = np.array(treinamento)

#Separando os dados
train_x = list(treinamento[:, 0])
train_y = list(treinamento[:, 1])

#Criando um modelo de machine learning sequencial
modelo = Sequential()
modelo.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(64, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(len(train_y[0]), activation='softmax'))

#Compilando o modelo
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
modelo.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = modelo.fit(np.array(train_x), np.array(train_y), epochs=15000, batch_size=64, verbose=1)

#Salvando o modelo
modelo.save("chatbotmodel.h5", hist)
print("Tudo certo!")