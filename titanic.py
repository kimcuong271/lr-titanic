import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics

df = pd.read_csv('train.csv')

#Data preprocessing
df['Sex']=df['Sex'].map(lambda x: 0 if x=='male' else 1)
df = df[['Age','Sex','Pclass','SibSp','Parch','Fare','Survived']]
df = df.dropna()
X= df.drop(['Survived'],axis=1)
y = df['Survived']

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
scaler = StandardScaler()
train_features=scaler.fit_transform(X_train)
test_features = scaler.fit_transform(X_test)

#Build Model
model = LogisticRegression()
model.fit(X_train,y_train)
# Evaluation
train_score = model.score(train_features,y_train)
test_score = model.score(test_features,y_test)
y_hat = model.predict(test_features)
confusion = metrics.confusion_matrix(y_test,y_hat)
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]
metrics.classification_report(y_test,y_hat)
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_hat)
auc = metrics.roc_auc_score(y_test,y_hat)
# P2
st.title("Data Science")
st.write("## Titanic Survial Prediction Project")
menu = ['Overview','Build Project','New Prediction']
choice = st.sidebar.selectbox('Menu',menu)
if choice == "Overview":
    st.subheader('Overview')
    st.write("""
      #### The data has been split into two groups:
    - training set (train.csv):
    The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.
    - test set (test.csv):
    The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.
    - gender_submission.csv:  a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.

    """
    )
elif choice =='Build Project':
    st.subheader("Build Project")
    st.write("### Data Processing")
    st.write("""#### Show Data:""")
    st.table(df.head(5))
    st.write("#### Build Model and Evaluation")
    st.write("Train Set Score: {}".format(round(train_score,2)))
    st.write("Test Set Score: {}".format(round(test_score,2)))
    st.write("Confusion Matrix: ")
    st.table(confusion)
    st.write(metrics.classification_report(y_test,y_hat))
    st.write("### AUC: %.3f"% auc)
    st.write("#### Visualization")
    fig, ax = plt.subplots()
    ax.bar(['False Negative','True Negative','True Postive','False Positive'],[FN,TN,TP,FP])
    st.pyplot(fig)
    st.write("ROC Curve")
    fig1,ax1 =plt.subplots()
    ax1.plot([0,1],[0,1],linestyle='--')
    ax1.plot(fpr,tpr,marker='.')
    ax1.set_title("ROC CURVE")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    st.pyplot(fig1)
elif choice =="New Prediction":
    st.subheader("Make New Prediction")
    st.write("### Input/Select Data")
    name = st.text_input("Name Of Passanger")
    sex = st.selectbox("Sex",options=["Male","Female"])
    age = st.slider("Age",1,100,1)
    Pclass = np.sort(df['Pclass'].unique())
    pclass =st.selectbox("Pclass",options=Pclass)
    max_sib = max(df['SibSp'])
    sibsp= st.slider("Siblings",0,max_sib,1)
    max_parch = max(df['Parch'])
    parch = st.slider("Parch",0,max_parch,1)
    max_fare = round(max(df['Fare'])+10,2)
    fare=st.slider("Fare",0.0,max_fare,0.1)
    sex=0 if sex =="Male" else 1
    new_data =  scaler.transform([[sex,age,pclass,sibsp,parch,fare]])
    prediction = model.predict(new_data)
    predict_prob = model.predict_proba(new_data)
    print(prediction[0])
    if prediction[0] == 1:
        st.subheader("Passenger {} would survived with probability of {}%".format(name,round(predict_prob[0][1]*100,2)))
    else:
        st.subheader("Passenger {} would not survived with probability of {}%".format(name,round(predict_prob[0][1]*100,2)))

    