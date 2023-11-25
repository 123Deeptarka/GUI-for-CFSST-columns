#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 19:41:45 2023

@author: deeptarkaroy
"""

import streamlit as st
import pandas as pd 
import shap 
import matplotlib.pyplot as plt


st.write("""
 # Graphical User Interface for Circular CFSST Columns:"""
        )

st.write('---')

df=pd.read_excel(r"/Users/deeptarkaroy/Desktop/test.xlsx")

x=df.drop(["N_Test"],axis=1)
y=df["N_Test"]


st.sidebar.header("User Input Parameters:")

def user_input_features():
    Material=st.sidebar.slider("Grade_SS",1,6,3)
    D=st.sidebar.slider("D",50,325,150)
    t=st.sidebar.slider("t",1,12,3)
    L=st.sidebar.slider("L",150,977,377)
    LB=st.sidebar.slider("L/D",2.36,6.02,3.16)
    Eo=st.sidebar.slider("Eo",173900,217000,199730)
    f=st.sidebar.slider("f_0.2",228,544,433)
    fu=st.sidebar.slider("fu",539,786,674)
    n=st.sidebar.slider("n",3.0,10.0,5.97)
    fc=st.sidebar.slider("fc",20.0,144.4,48.14)
    data={"Grade_SS":Material,"D":D,"t":t,"L":L,"L/D":LB,"Eo":Eo,"f_0.2":f,"fu":fu,"n":n,"fc":fc}
    features=pd.DataFrame(data,index=[0])
    
    return features

data_df=user_input_features()

st.header("Input Data:")
st.write(data_df)

st.write("---")

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


from catboost import CatBoostRegressor,Pool
model=CatBoostRegressor(n_estimators=700,learning_rate=0.1)
model.fit(x,y)

prediction=model.predict(data_df)


st.header(" Predicted Axial Capacity of Columns(KN):")
st.write(prediction)
st.write('---')



explainer=shap.TreeExplainer(model)
shap_values=explainer.shap_values(x)

#st.header("Feature Importance")
plt.title("Feature Importance Based on Shap Values")
fig,ax=plt.subplots()
ax=shap.summary_plot(shap_values,x)
#st.pyplot(fig)
#st.set_option('deprecation.showPyplotGlobalUse', False)
#st.write("---")

plt.title("Relative Feature Importance")
shap.summary_plot(shap_values,x,plot_type="bar")
#st.pyplot(bbox_inches="tight")






