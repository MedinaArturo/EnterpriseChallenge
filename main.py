import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import gradio as gr

lematizador = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
palavras = pickle.load(open('palavras.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
modelo = load_model('chatbotmodel.h5')


def limpar_frases(frase):
    palavras_frase = nltk.word_tokenize(frase)
    palavras_frase = [lematizador.lemmatize(word)
                      for word in palavras_frase]
    return palavras_frase


def bagw(frase):
    palavras_frase = limpar_frases(frase)
    bag = [0] * len(palavras)
    for w in palavras_frase:
        for i, word in enumerate(palavras):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predicao_classe(frase):
    bow = bagw(frase)
    res = modelo.predict(np.array([bow]))[0]
    error_threshold = 0.25
    resultados = [[i, r] for i, r in enumerate(res)
                  if r > error_threshold]
    resultados.sort(key=lambda x: x[1], reverse=True)
    lista_retorno = []
    for r in resultados:
        lista_retorno.append({'intent': classes[r[0]], 'probability': str(r[1])})
        return lista_retorno


def resposta(lista_intents, json_intents):
    tag = lista_intents[0]['intent']
    lista_de_intents = json_intents['intents']
    resultado = ""
    for i in lista_de_intents:
        if i['tag'] == tag:
            resultado = random.choice(i['responses'])
            break
    return resultado


print("Chatbot lançado.")

with gr.Blocks(theme=gr.themes.Soft(), title="Allyx - Chatbot") as demo:
    chatbot = gr.Chatbot(label="Allyx")
    msg = gr.Textbox(label="Usuário", placeholder="Digite a sua mensagem")
    clear = gr.Button("Clear")


    def main(mensagem, historico=[]):
        ints = predicao_classe(mensagem)
        res = resposta(ints, intents)
        historico.append((mensagem, res))
        return "", historico


    msg.submit(main, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    demo.launch(share=True)
