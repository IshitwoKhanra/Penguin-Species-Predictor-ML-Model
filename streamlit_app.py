import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title("Penguin Species Predictor")

st.info("This project predicts the species of penguins using various features such as Island, Bill length(mm), Bill depth(mm),Flipper length(mm), Body mass(g) and Gender")

with st.expander('Data'):
    st.write('**Raw data**')
    df=pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    df
    
    st.write("**X**")
    X_raw=df.drop('species',axis=1)
    X_raw

    st.write('**y**')
    y_raw=df.species
    y_raw

with st.expander('Data Visualisation'):
    st.scatter_chart(data=df,x='bill_length_mm',y='body_mass_g',color='species')

#Input features
with st.sidebar:
    st.header('Input features')

    island=st.selectbox("Island",('Torgersen','Biscone','Dream'))
    bill_length_mm=st.slider("Bill Length(mm)", 32.1,59.6,43.9)
    bill_depth_mm=st.slider('Bill depth(mm)',13.1,21.5,17.2)
    flipper_length_mm=st.slider('Flipper length(mm)',172.0,231.0,201.0)
    body_mass_g=st.slider('Body mass (g)',2700.0,6300.0,4207.0)
    gender=st.selectbox("Gender",('Male','Female'))

    #Create dataframe for the input features
    data={'island':island,
        'bill_length_mm':bill_length_mm,
        'bill_depth_mm':bill_depth_mm,
        'flipper_length_mm':flipper_length_mm,
        'body_mass_g':body_mass_g,
        'sex':gender}

    input_df=pd.DataFrame(data,index=[0])
    input_penguins=pd.concat([input_df,X_raw],axis=0)

#Whenever the user chooses a value it will be reflected in the dataset
with st.expander('Input features'):
    st.write('**Input penguins(based on slider)**')
    input_df
    st.write('**Combined data**')
    input_penguins

#Data Prep
#Encode x
encode=['island','sex']
df_penguins=pd.get_dummies(input_penguins,prefix=encode)

X=df_penguins[1:] #Dataset X for training the model later
input_row=df_penguins[:1]

#Encode y
target_mapper={'Adelie':0,
               'Chinstrap':1,
               'Gentoo':2}
    
def target_encode(val):
    return target_mapper[val]
    
y=y_raw.apply(target_encode)

with st.expander('Data Preparation'):
    st.write('**Encoded X (Input Penguin)**')
    input_row
    st.write('**Encoded y**')
    y

#Model training and Inference
##Train the ML model
clf=RandomForestClassifier()
clf.fit(X,y)

##Apply the model to make predictions
prediction=clf.predict(input_row)
predictions_proba=clf.predict_proba(input_row)

# df_predictions_proba=pd.DataFrame(predictions_proba)

# df_predictions_proba.columns={'Adelie','Chinstrap','Gentoo'}

# df_predictions_proba.rename(columns={0:'Adelie',
#                                      1:'Chinstrap',
#                                      2:'Gentoo'})

df_predictions_proba = pd.DataFrame(predictions_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])


#Displaying the predicted species

st.subheader("Predicted Species")
st.dataframe(df_predictions_proba,
             column_config={
                'Adelie':st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
                ),
                'Chinstrap':st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
                ),
                'Gentoo':st.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
                )
             },hide_index=True)


penguin_species=np.array(['Adelie','Chinstrap','Gentoo'])
st.success(str(penguin_species[prediction][0]))