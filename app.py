import pandas as pd
import streamlit as st
import pickle
import tensorflow.keras.models as load_models

st.title("Titanic Survival Prediction")

pclass = st.slider("Enter the passenger class for the user",1,3)
sex = st.selectbox('Enter the passenger gender', ['male', 'female'])
sibsp = st.slider("Enter the passenger number of siblings/spouses aboard the Titanic",0,8)
parch = st.slider("Enter the passenger number of parents/children aboard the Titanic",0,6)
fare = st.number_input("Enter the passenger fare", min_value=0.0, step=0.01)
embarked = st.selectbox('Enter the passenger port of embarkation', ['Southampton', 'Cherbourg', 'Queenstown'])

data = {'Pclass': pclass, 'Sex': sex, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked}

model = load_models.load_model("model.h5")

with open('label_encoder.pkl', 'rb') as f:
    label = pickle.load(f)

with open('onehot_encoder.pkl', 'rb') as f:
    onehot = pickle.load(f)

with open('scale.pkl', 'rb') as f:
    scale = pickle.load(f)

data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])
data['Sex'] = label.transform(data['Sex'])
embarked = onehot.transform(data[['Embarked']])
embarked_df = pd.DataFrame(embarked, columns=onehot.get_feature_names_out())
data = pd.concat(
        [data.drop(columns=["Embarked"]), embarked_df],
        axis=1
    )

data[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']]=scale.transform(data[['Pclass','Sex', 'SibSp' , 'Parch', 'Fare']])

y=model.predict(data)
y=y[0][0]

def Chance(y):
    if y>=0.5:
        return "The passenger is predicted to survive."
    else:
        return "The passenger is predicted not to survive."

if st.button('Predict of survival Chance'):
    st.write('Probability of survival:', y)
    st.write(Chance(y))