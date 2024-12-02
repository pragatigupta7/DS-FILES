#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pickle
import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


# In[17]:


model= pickle.load(open('l28.pkl','rb'))


# In[18]:


st.title('Model Deployment using Logistic Regression')


# In[20]:


def user_input_parameters():
    CLMSEX= st.sidebar.selectbox('Gender, Male-1,Female-0',[0,1])
    CLMINSUR= st.sidebar.selectbox('Insurance, Yes-1,No-0',[0,1])
    SEATBELT= st.sidebar.selectbox('Seatbelt, Yes-1,No-0',[0,1])
    CLMAGE=st.sidebar.slider('Age',0,50)
    LOSS= st.sidebar.slider('Loss',0.100)
    data= {'CLMSEX':CLMSEX,'CLMINSUR':CLMINSUR,'SEATBELT':SEATBELT,'CLMAGE':CLMAGE,'LOSS':LOSS}
    features= pd.DataFrame(data,index=[0])
    return features
df= user_input_parameters()
st.subheader('User Input Parameters')
st.write(df)
pred= model.predict(df)
pred_prob= model.predict_proba(df)
st.subheader('Predicted Value')
st.write('Yes' if pred_prob[0][1]>+0.5 else 'No')
st.subheader('Predicted_proba')
st.write(pred_prob)


# In[ ]:




