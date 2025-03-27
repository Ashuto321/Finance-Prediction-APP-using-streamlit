import streamlit as st
import pandas as pd
import matplotlib.pyplot
from matplotlib import pyplot as plt
from plotly import graph_objs as go 
from sklearn.linear_model import LinearRegression
import numpy as np

data=pd.read_csv("data//123.csv")

#machine learning model
#filling nan values 
data["Years of Experience"].fillna(data["Years of Experience"].mean(), inplace=True)
data["Salary"].fillna(data["Salary"].mean(), inplace=True)
#converting into numpy array
x=np.array(data['Years of Experience']).reshape(-1,1)
y=np.array(data['Salary'])
#training model
lr = LinearRegression()
lr.fit(x,y)


st.title("Salary Predictor")

nav=st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])

if nav=="Home":
       st.image("data//nemo.jpg")
       if st.checkbox("Show Table"):
              st.table(data)
       
       graph=st.selectbox("what kind of Graph?",["Non-Interactive","Interactive"])
       
       val= st.slider("Filter your fetaures uisng age",0,20)
       data = data.loc[data["Years of Experience"]>=val]#to make your filter function

       if graph=="Non-Interactive":
              plt.figure(figsize=(10,5))
              plt.scatter(data["Years of Experience"],data["Salary"])#writing on x axis and y axis
              plt.ylim(0)
              plt.xlabel("Years of Experience")
              plt.ylabel("Salary")
              plt.tight_layout()
              st.pyplot()
       if graph =="Interactive":
              layout=go.Layout(
                     xaxis=dict(range=[0,16]),
                     yaxis= dict(range=[0,210000])
              )
              fig=go.Figure(data=go.Scatter(x=data["Years of Experience"],y=data["Salary"], mode='markers'),layout = layout)
              st.plotly_chart(fig)

if nav=="Prediction":
       st.header("Know your salary")
       val= st.number_input("Enter your exp",0.00,20.00,step=0.25)
       val=np.array(val).reshape(1,-1)
       pred = lr.predict(val)[0]

       if st.button("Predict"):
              st.success(f"your predictied salary is {round(pred)}")

if nav=="Contribute":
       st.header("Contribe to our dataset")
       ex=st.number_input("Enter your experience",0.0,20.0)
       sal= st.number_input("Enter your salary",0.0,1000000.0, step=1000.0)
       if st.button("Submit"):
              to_add={"Years of Experience": ex, "Salary":sal}
              to_add=pd.DataFrame([to_add])
              to_add.to_csv("data//123.csv", mode='a', header=False, index= False)
              st.success("Submitted")
