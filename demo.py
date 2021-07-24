import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
df= pd.read_csv("train.csv")
st.title("Trung Tâm Tin Học")
st.header("Data Science")
menu = ["Display Text","Display Data","Display Chart","Display Interactive Widget"]
choice = st.sidebar.selectbox("Menu",menu)
if choice =="Display Text":
    st.subheader("Hành Trang Tốt Nghiệp Data Science")
    st.text("Khoá học được thiết kế nhằm bổ sung kiến thức cho học viên")
    st.markdown("### Có 5 Chủ Đề")
    st.write("""
    - Chủ đề 1
    - Chủ đề 2
    - ...
    """)

    st.markdown("### Ngôn Ngữ Lập Trình Python")
    st.code("st.display_text_function('Nội Dung')",language='python')
elif choice =="Display Data":
    st.write("### Display Data")
    st.dataframe(df.head(3))
    st.table(df.head(3))
    st.json(df.head(10).to_json())
elif choice =="Display Chart":
    st.write("### Display Chart")
    count_Pclas = df[['PassengerId','Pclass']].groupby(['Pclass']).count()
    st.bar_chart(count_Pclas)
    fig,ax = plt.subplots()
    ax = sns.boxplot(x='Pclass',y='Fare',data=df)
    st.pyplot(fig)
else:
    st.write("## Display Interactive Widget")
    st.write("### Input your Data")
    name = st.text_input("Name")
    sex = st.radio("Sex",options=['Male','Female'])
    age = st.slider("Age",1,100)
    jobtime = st.selectbox("You have",options=['Part Time Job','Full Time Job'])
    house =st.checkbox("Have House/Apartment")
    hobbies = st.multiselect("Hobbies",options=['Cooking','Reading','Writing','Travel','Others'])
    submit = st.button("Submit")
    if submit:
        st.write("### Your infomation:")
        st.write(" Name: ",name,
        " - Sex: ",sex,
        " - Age: ",age,
        " - Job Time: ", jobtime,
        " - You have a ","House " if house else "", 
        " - Hobbies: ",", ".join(map(str,hobbies)))