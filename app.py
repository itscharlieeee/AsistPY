
import os
#from dotenv import load_dotenv
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
#from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
import streamlit as st
import json
import pandas as pd
import numpy as np
from PIL import Image

st.title('Analítica de datos con Agentes 🤖🔍')
image = Image.open('data_analisis.png')
st.image(image,width=350)

with st.sidebar:
   st.subheader("Este Agente de Pandas, te ayudará a realizar algo de análisis sobre tus datos")

ke = st.text_input('Ingresa tu Clave')
os.environ['OPENAI_API_KEY'] = ke


uploaded_file = st.file_uploader('Elige el archivo csv')
if uploaded_file is not None:
   df=pd.read_csv(uploaded_file, on_bad_lines='skip') 
   st.write(df)

st.subheader('Te ayduaré a analizar los datos que cargues.')

user_question = st.text_input("Que desesas saber de los datos?:")
if user_question :
    user_question=user_question+', busca  primero siempre la correspondencia entre las columnas y la información que te pida'
  #try:

        #if user_question:
    agent = create_pandas_dataframe_agent(OpenAI(model_name="gpt-4o-mini",temperature=0, max_tokens=1500), df,allow_dangerous_code=True)
    with get_openai_callback() as cb:
        response = agent.run(user_question)
       #print(cb)
    st.write(response)
  #except:
   # pass    
