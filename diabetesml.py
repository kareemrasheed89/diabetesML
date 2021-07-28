#Building a diabetes detection webapp
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st


st.header("""Diabetes Detection: Detect if someone have diabetes or not using ML model""")
image=Image.open("C:/Users/user/Desktop/diabetesML/ML.jpg")
st.image(image, caption='ML',use_column_width=True)
df=pd.read_csv("C:/Users/user/Desktop/diabetesML/Diabetesdata.csv")
data_metrics=df.describe()
st.dataframe(df)
st.write(data_metrics)
chart=st.bar_chart(df.head(10))
X=df.iloc[:,0:8].values
Y=df.iloc[:,-1].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25, random_state=0)


def get_user_input():
    pregnancies=st.sidebar.slider('pregancies', 0,10,2)
    glucose=st.sidebar.slider('glucose', 0,100,20)
    blood_pressure=st.sidebar.slider('BP', 0,120,40)
    skin_thickness=st.sidebar.slider('skin_tickness', 0,150,50)
    insulin=st.sidebar.slider('insulin', 0,100,25)
    bmi=st.sidebar.slider('bmi', 0,100,30)
    dpf=st.sidebar.slider('dpf', 0,1,0)
    Age=st.sidebar.slider('Age', 0,100,5)

    user_data={
'pregancies':pregnancies,
'glucose':glucose,
'blood_pressure':blood_pressure,
'skin_tickness':skin_thickness,
'insulin':insulin,
'bmi':bmi,
'dpf':dpf,
'age':Age
}

    features=pd.DataFrame(user_data,index=[0])
    return features
user_input=get_user_input()
st.subheader("user input:")
st.write(user_input)

RandomForestClassifier=RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)

prediction=RandomForestClassifier.predict(user_input)
#st.subheader('Model Accuracy Score')
#st.write(mean_squared_error(X_test,prediction))

st.subheader('Classification:')
st.write(prediction)

if prediction ==1:
    st.warning("This person shows a sign of having DIABETES")
elif prediction==0:
    st.info("This patient is free from DIABETES")

image2=Image.open('C:/Users/user/Desktop/diabetesML/Hazontech.png')
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.image(image2, caption='hazontech', use_column_width=False)
st.write("Rasheed Kareem(Chief Data Officer)")
