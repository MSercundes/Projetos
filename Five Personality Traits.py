#!/usr/bin/env python
# coding: utf-8

# # Five Personality Traits (OCEAN)

# #### Importando as bibliotecas

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import open 
pd.options.display.max_columns = 150
import csv 


# #### Carregando o Dataset

# In[42]:


data = pd.read_csv('data-final.csv', sep='\t')


# #### Verificando o Dataset

# In[43]:


data.head()


# #### Excluindo os atributos irrelevantes

# In[44]:


data.drop(data.columns[50:110], axis=1, inplace=True)


# #### Verificando novamente os dados, agora só com as perguntas de fato

# In[45]:


data.head()


# #### Analisando estatísticas da base de dados

# In[46]:


pd.options.display.float_format = "{:.2f}".format 
data.describe()


# #### Verificando a quantidade de registro por valor

# In[47]:


data["EXT1"].value_counts()


# #### Selecionando o total de registros com valor zero

# In[48]:


data[(data == 0.00).all(axis=1)].describe()


# #### Limpando o Dataframe com apenas registros maiores que zero

# In[49]:


data = data[(data > 0.00).all(axis=1)]


# #### Verificando a contagem de registros por valor 

# In[50]:


data["EXT1"].value_counts()


# ### Qual o número de clusters que vamos definir?

# #### Instalando a yellowbrick

# In[51]:


get_ipython().system('pip install yellowbrick')


# #### Importando bibliotecas 

# In[52]:


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


# #### Iniciando o médoto KMeans e o Visualizer 

# In[53]:


kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,10))


# #### Selecionando uma amostra aleatória dos dados com 5000 observações

# In[54]:


data_sample = data.sample(n=5000, random_state=1)


# ### Executando o teste

# In[55]:


visualizer.fit(data_sample)
visualizer.poof()


# ### Agrupando os participantes em 5 grupos

# #### Atribuindo os registros aos devidos grupos

# In[56]:


kmeans = KMeans(n_clusters=5)
k_fit = kmeans.fit(data)


# #### Inserindo os rótulos dos clusters no dataframe

# In[57]:


predicoes = k_fit.labels_
data['Clusters'] = predicoes


# #### Verificando os dados

# In[58]:


data.head()


# ### Analisando os grupos

# #### Qual a quantidade de observações em cada grupo?

# In[59]:


data["Clusters"].value_counts()


# #### Agrupando o registro por grupos

# In[60]:


data.groupby('Clusters').mean()


# #### Calculando a média de cada grupo de questões para verificar um padrão

# #### Selecionando as colunas de cada grupo

# In[61]:


col_list = list(data)
ext = col_list[0:10]
est = col_list[10:20]
agr = col_list[20:30]
csn = col_list[30:40]
opn = col_list[40:50]


# #### Somando os valores de cada grupo

# In[62]:


data_soma = pd.DataFrame()
data_soma['extroversion'] = data[ext].sum(axis=1)/10
data_soma['neurotic'] = data[est].sum(axis=1)/10
data_soma['agreeable'] = data[agr].sum(axis=1)/10
data_soma['conscientious'] = data[csn].sum(axis=1)/10
data_soma['open'] = data[opn].sum(axis=1)/10
data_soma['clusters'] = predicoes


# #### Exibindo o valor médio por grupo

# In[63]:


data_soma.groupby('clusters').mean()


# #### Visualizando as médias por grupo

# In[64]:


data_clusters = data_soma.groupby('clusters').mean()


# In[65]:


plt.figure(figsize=(22,3))
for i in range(0, 5):
    plt.subplot(1,5,i+1)
    plt.bar(data_clusters.columns, data_clusters.iloc[:, i], color='blue', alpha=0.2)
    plt.plot(data_clusters.columns, data_clusters.iloc[:, i], color='red')
    plt.title('Grupo ' + str(i))
    plt.xticks(rotation=45)
    plt.ylim(0,4);


# #### Instalando a biblioteca gradio

# In[70]:


get_ipython().system('pip install gradio')


# In[71]:


import gradio as gr


# #### Lendo os dados com as questões

# In[72]:


dicio_questions = open("questions.txt").read().split("\n")


# #### Verificando os dados

# In[73]:


dicio_questions


# In[74]:


questions = []
for q in dicio_questions:
    q = str(q)
    questions.append(q[q.find("\t"):].lstrip())


# In[75]:


questions


# #### Criando os inputs dinamicos para passar ao gradio

# In[78]:


inputs_questions = []
for q in questions:
  obj_input = gr.inputs.Slider(minimum=1,maximum=5,step=1,default=3,label=q)
  inputs_questions.append(obj_input)


# #### Verificando os inputs

# In[77]:


inputs_questions


# In[79]:


def predict(*outputs_questions):
    outputs_questions = np.array(outputs_questions).reshape(1, -1)
    return k_fit.predict(outputs_questions)

iface = gr.Interface(
                    fn = predict,
                    title = "Big Five Personality",
                    description = "Sistema para detecção de traços de personalidade.",
                    inputs = inputs_questions,
                    outputs="text")
iface.launch(share=True)


# In[ ]:




