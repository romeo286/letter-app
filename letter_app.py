import streamlit as st
import os
import joblib
from PIL import Image

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

@st.cache
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df





def load_prediction_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model





def main():
    st.title("Letter Classification Using Naive Bayes")
    st.subheader("A Classification Problem")

    menu = ["EDA","Prediction","About_Team"]
    choices = st.sidebar.selectbox("Select Menu",menu)

    if choices == 'EDA':
        st.subheader("EDA\Exploratory Data Analysis ")

        # st.subheader("Dataset")
        st.success("Dataset")  
        data = load_data('data/lrd.csv')
        st.dataframe(data.head(10))

        if st.checkbox("Show Summary"):
            st.write(data.describe())

        if st.checkbox("Show Null Values"):
            st.write(data.isnull())

        if st.checkbox("Show Shape"):
            st.write(data.shape)    

        if st.checkbox("Show whether the given dataset is balanced or not"):
            st.write(data['lettr'].value_counts().plot(kind='bar'))
            st.pyplot()

        if st.checkbox("Pie Chart"):
            st.write(data['lettr'].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()

        st.subheader("Required Libaries")
        st.success("Import all this libaries")

        st.code("import numpy as np")
        st.code("import pandas  as pd")  
        st.code("import matplotlib.pyplot as plt")  
        st.code("import os")
        st.code("from PIL import Image")
        st.code("import joblib")
        # st.code("pip install sklearn")
        st.code("import streamlit as st")
        

        st.subheader("Required Enviroments")
        st.success("Enviroment Requirements")
        data = {
            "Python-version" : "3.6",
            "Platform" : "Anaconda\Jupyter Notebook\Streamlit",
            "Code-editor" : " VS-Code",
            "Deployment-Server" : "Heroku"
        }

        st.json(data)     





# x-box,ybox,width,high,onpix,x-bar,y-bar,x2bar,y2bar,xybar,x2ybr,xy2br,x-ege,xegvy,y-ege,yegvx


    if choices == 'Prediction':
        st.success("Prediction")  

        x_box = st.number_input("Select x_box attributes between 1 to 12",1,12)
        ybox =  st.number_input("Select ybox attributes between 1 to 12",1,12)
        width = st.number_input("Select width attributes between 1 to 12",1,12)
        high = st.number_input("Select high attributes between 1 to 12",1,12)
        onpix = st.number_input("Select onpix attributes between 1 to 12",1,12)
        x_bar = st.number_input("Select x_bar attributes between 1 to 12",1,12)
        y_bar = st.number_input("Select y_bar attributes between 1 to 12",1,12)
        x2bar = st.number_input("Select x2bar attributes between 1 to 12",1,12)
        y2bar = st.number_input("Select y2bar attributes between 1 to 12",1,12)
        xybar = st.number_input("Select xybar attributes between 1 to 12",1,12)
        x2ybr = st.number_input("Select x2ybar attributes between 1 to 12",1,12)
        xy2br = st.number_input("Select xy2br attributes between 1 to 12",1,12)
        x_ege = st.number_input("Select x_ege attributes between 1 to 12",1,12)
        xegvy = st.number_input("Select xegvy attributes between 1 to 12",1,12)
        y_ege = st.number_input("Select y_ege attributes between 1 to 12",1,12)
        yegvx = st.number_input("Select yegvx attributes between 1 to 12",1,12)




        pretty_data = {
            "x_box" : x_box,
            "ybox" : ybox,
            "width" : width,
            "high" : high,
            "onpix" : onpix,
            "x_bar" : x_bar,
            "y_bar" : y_bar,
            "x2bar" : x2bar,
            "y2bar" : y2bar,
            "xybar" : xybar,
            "x2ybr" : x2ybr, 
            "xy2br" : xy2br,
            "x_ege" : x_ege,
            "xegvy" : xegvy,
            "y_ege" : y_ege,
            "yegvx" : yegvx,
            
        }

        st.subheader("Options Selected")
        st.json(pretty_data)







        st.subheader("Data Encoded As")
        sample_data =[x_box,ybox,width,high,onpix,x_bar,y_bar,x2bar,y2bar,xybar,x2ybr,xy2br,x_ege,xegvy,y_ege,yegvx]
        st.write(sample_data)

        prep_data = np.array(sample_data).reshape(1,-1)

        model_choice = st.selectbox("Model Choice",["NaiveBayes"])
        if st.button("Evaluate"):
            if model_choice == 'NaiveBayes':

                predictor = load_prediction_model("models/model_joblib.pkl")
                prediction =predictor.predict(prep_data)
                st.success("Result") 
                st.write(prediction)




    
    
    
    
    
    
    
    
    if choices == 'About_Team':
        st.success("About Us")  
         
        st.subheader("Project Co-ordinator")

        image = Image.open('img/sir.jpg')
        st.image(image,width=300,caption="Mr. Rabinder Kr Prasad") 
        # st.text('Rabinder Kumar Prasad')
        image = Image.open('img/mam.jpg')
        st.image(image,width=300,caption="Ms. Charu Guleria")
        # st.text(' Charu Guleria')

        st.subheader("Project Team-mates")


        image = Image.open('img/bijit.jpg')
        st.image(image,width=300,caption="Bijit Phukan")
        # st.text('Bijit Phukan')
        
        image = Image.open('img/admand.jpg')
        st.image(image,width=300,caption="Admandu Saikia")
        # st.text('Admandu Saikia')
        image = Image.open('img/Arup.jpg')
        st.image(image,width=300,caption="Arup Kumar Dey")
        # st.text('Arup Kumar Dey')        






if __name__ == '__main__':
	main()

