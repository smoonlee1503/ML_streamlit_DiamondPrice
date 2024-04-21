#!/usr/bin/env python
# coding: utf-8



# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#title
st.title("Data Science Project using Streamlit")

# In[7]:


df = sns.load_dataset('diamonds')


# In[15]:


menu = st.sidebar.radio("Menu",["Home","Prediction of Price"])
if menu=="Home":
    st.header("Diamond Price Analysis")
    st.image("diamonds.jpeg")
    st.header("Explonatory Data Analysis")
    if st.checkbox("Tabular Data"):
        st.subheader("Diamond Price Tabular Data")
        df
        st.write("Shape of dataset", df.shape)
    if st.checkbox("Statistical Summary"):
        st.subheader("Statistical Summary of Dataframe")
        st.table(df.describe())
        st.markdown("Correlation graph")
        fig,ax = plt.subplots(figsize=(5,2.5))
        sns.heatmap(df.drop(["cut","color","clarity"],axis=1).corr(), annot=True, cmap="coolwarm")
        st.pyplot(fig)
    if st.checkbox("Graph"):
        st.subheader("Visualization of Dataframe")
        graph=st.selectbox("Different graphs",["Scatter Plot", "Bar Graph","Histogram"])
        if graph=="Scatter Plot":
            value=st.slider("Filter data using carat",0,6)
            df = df.loc[df["carat"]>=value]
            fig,ax = plt.subplots(figsize=(10,5))
            sns.scatterplot(data=df, x="carat",y="price",hue="cut")
            st.pyplot(fig)
        if graph=="Bar Graph":
            fig,ax = plt.subplots(figsize=(3.5,2))
            sns.barplot(x="cut",y=df.cut.index, data=df)
            st.pyplot(fig)
        if graph=="Histogram":
            fig,ax = plt.subplots(figsize=(5,3))
            sns.distplot(df.price,kde=True)
            st.pyplot(fig)

if menu=="Prediction of Price":
    st.header("Diamond Price Prediction")
    st.image("DiamondPrice.jpg")
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    X = np.array(df['carat']).reshape(-1,1)
    y = np.array(df['price']).reshape(-1,1)
    lr.fit(X,y)
    value=st.number_input("Carat",0.20,5.0,step=0.15)
    value=np.array(value).reshape(-1,1)
    prediction=lr.predict(value)[0]
    if st.button("Price Prediction ($)"):
        st.write(f"{prediction}")






# In[ ]:




