

import streamlit as st
import pandas as pd
from pysentimiento import create_analyzer
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def ai_soledad(df_bot):
  analyzer = create_analyzer(task="emotion", lang="es")

  enfado = []
  asco = []
  miedo = []
  alegria = []
  tristeza = []
  sorpresa = []

  dataframe = pd.read_csv('datos_serena_mod.csv', encoding="ISO-8859-1")
  

  sexo = dataframe['sexo'].tolist()
  edad = dataframe['edad'].tolist()
  vive_solo = dataframe['vive_solo'].tolist()
  comentarios = dataframe['comentario'].tolist()

  estado_soledad = dataframe['hay_soledad'].tolist()

  emotions = analyzer.predict(comentarios)

  for analizer_output in emotions:
    emotions_scores = analizer_output.probas
    enfado.append(emotions_scores['anger'])
    asco.append(emotions_scores['disgust'])
    miedo.append(emotions_scores['fear'])
    alegria.append(emotions_scores['joy'])
    tristeza.append(emotions_scores['sadness'])
    sorpresa.append(emotions_scores['surprise'])

  dataframe['sexo'] = sexo
  dataframe['edad'] = edad
  dataframe['vive_solo'] = vive_solo
  dataframe['comentario'] = comentarios
  dataframe['hay_soledad'] = estado_soledad
  dataframe['enfado'] = enfado
  dataframe['asco'] = asco
  dataframe['miedo'] = miedo
  dataframe['alegria'] = alegria
  dataframe['tristeza'] = tristeza
  dataframe['sorpresa'] = sorpresa

  y = dataframe['hay_soledad']
  X = dataframe.drop(columns=['Unnamed: 0', 'sexo','edad','vive_solo','comentario','hay_soledad'])

  
  estimadores = 25
  depth = [5,10,15]

  scores = list()
  mejor_RFC = [0, 0, 0]
  mejor_KNN = [0, 0]

  #print ('RANDOMFORESTCLASSIFIER')
  for i in range (1,estimadores):
    for j in depth:
      #print('Estimador: ',i,' Depth: ', j)
      model = RandomForestClassifier(n_estimators=i, max_depth=j)
      result = cross_val_score(model, X, y, cv=3)
      scores.append(result)
      if result.mean() > mejor_RFC[2]:
        mejor_RFC = [i, j, result.mean()]

  #print ('kNEIGHBORSCLASSIFIER')
  for i in range (1,10):
      #print('Vecinos: ',i,)
      model = KNeighborsClassifier(n_neighbors=i)
      result = cross_val_score(model, X, y, cv=3)
      scores.append(result)
      if result.mean() > mejor_KNN[1]:
        mejor_KNN = [i, result.mean()]



  model_AIsolation = RandomForestClassifier(n_estimators=mejor_RFC[1], max_depth=mejor_RFC[2])
  model_AIsolation.fit(X, y)

  ## Obtener resultado a partir de comentario nuevo
  emotions = analyzer.predict(df_bot['comentario'].tolist())

  lista_sentimientos = ['enfado','asco','miedo','alegria','tristeza','sorpresa']
  list_sentiments = ['anger', 'disgust','fear', 'joy','sadness','surprise']
  for i,sent in enumerate(lista_sentimientos):
    df_bot.insert(len(df_bot.columns),sent,emotions[0].probas[list_sentiments[i]])
  X1 = df_bot.drop(columns=['sexo','edad','vive_solo','comentario'])
  soledad = model_AIsolation.predict(X1)

  return soledad






st.title("¡Bienvenido a AIsolation!")
st.write("Vas a formar parte de un proyecto para comprobar el estado de soledad en las personas a través de sus textos.")
inicio_choice = st.radio("Escoge uno: ",["Empecemos!","No me interesa"], index=1)
if inicio_choice == "No me interesa":
  st.write("Lamentamos que te vayas sin responder la encuesta.Si en algún momento te apetece responderla, puedes hacerlo accediendo de nuevo a la página.")

else:
  st.write("¡Muchas gracias por participar en el estudio! Vas a tener que responder a  3 preguntas. Tranquil@, te costará menos de 10 minutos.Por favor, respondelas con la mayor brevedad posible.")
  soledad_choice = st.radio("Escoge uno: ",["Ir a las preguntas","No me interesa"], index=1)

  if soledad_choice == "No me interesa":
     st.write("Lamentamos que te vayas sin responder la encuesta.Si en algún momento te apetece responderla, puedes hacerlo accediendo de nuevo a la página.")

  else: 
    st.write("Si tuvieras que contarme en un texto los sentimientos que has sentido esta semana. ¿Qué me contarías?")
    sentimientos_semana = st.text_area("Introduce tu respuesta")
    
    st.write("Para finalizar, me gustaria realizarte 3 preguntas de ámbito personal. Las preguntas serán: Género, Edad y con quien vives. ¿Estás de acuerdo?")
    personales_choice = st.radio("Escoge uno: ",["Si","No"], index=1)
    ## Falta terminar bot
    if personales_choice == "No":
      st.write("Lamentamos que te vayas sin responder la encuesta.Si en algún momento te apetece responderla, puedes hacerlo accediendo de nuevo a la página.")
    else:
      st.write("¿Vives solo?")
      solo_choice = st.radio("Escoge uno: ",["Si","No"], key = "Solo")
               
      st.write("¿Qué edad tienes?")
      edad_value = st.number_input("Entre 18 y 120", min_value=18, max_value=120)
               
      st.write("¿Con que género te identificas?")
      genero_choice = st.radio("Escoge uno: ",["Hombre","Mujer", "Otro", "Prefiero no contestar"], key = "Sexo")
      
      st.write("¡Muchas gracias por participar!")

      data = {'sexo': genero_choice, 'edad': edad_value, 'vive_solo': solo_choice, 'comentario': sentimientos_semana}
      df = pd.DataFrame(data=data, index=[0])

      soledad = -1

      if st.button("Obtener Resultado"):
       soledad = ai_soledad(df)
      
      if soledad==-1:
       st.write("Espere unos momentos...")
      elif soledad==0:
       st.write("Estas muy acompañado y no te sientes solo!")   
      else:
       st.write("Creo que deberías buscar ayuda profesional.")


      


      
