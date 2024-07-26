import streamlit as st
from chatbot import predict_class, get_response, intents
import re

st.title("Asistente virtual")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "first_message" not in st.session_state:
    st.session_state.first_message = True

# Función para convertir URLs y números de telefono y correos electrónicos a enlaces clicables
def convert_urls_to_links(response):
    # Convertir URLs
    response_with_links = response.replace('https://shuk.com.ar', '<a href="https://shuk.com.ar">shuk.com.ar</a>')
    response_with_links = response_with_links.replace('https://www.mercadopago.com.ar', '<a href="https://www.mercadopago.com.ar">mercadopago.com.ar</a>')


    # Convertir números de teléfono en enlaces de WhatsApp
    phone_numbers = re.findall(r'\+?\d{1,3}\d{10,14}', response)
    for number in phone_numbers:
        response_with_links = response_with_links.replace(number, f'<a href="https://wa.me/{number}">{number}</a>')

    # Expresión regular para detectar correos electrónicos
    #email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    #emails = email_pattern.findall(response)
    #for email in emails:
        #response_with_links = response_with_links.replace(email, f'<a href="mailto:{email}">{email}</a>')

    return response_with_links

for message in st.session_state.messages:
    with st.chat_message(message["role"]): #Para indicar el rol de aistente o usuario
        st.markdown(message["content"], unsafe_allow_html=True) #muestra el mensaje

if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("¡Hola!, ¿cómo puedo ayudarte?")
    st.session_state.messages.append({"role": "assistant", "content": "¡Hola!,¿cómo puedo ayudarte?"}) #Se añade al histórico
    st.session_state.first_message = False

if prompt := st.chat_input("¿cómo puedo ayudarte?"): # El texto es lo que aparecerá en el placeholder del prompt
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(({"role": "user", "content": prompt}))

    # IMPLEMENTACIÓN DEL ALGORITMO DE IA
    insts = predict_class(prompt)
    res = get_response(insts, intents)


    with st.chat_message("assistant"):
        # respuesta del chatbot (por ahora solo muestra lo mismo que escribe el cliente)
        # st.markdown(prompt)
        res_with_links = convert_urls_to_links(res)
        st.markdown(res_with_links.replace('\n', '\n\n'), unsafe_allow_html=True) # para tomar la respuesta de la ia
    # st.session_state.messages.append({ "role": "assistant", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": res_with_links})
