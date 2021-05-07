import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn import svm

@st.cache
def file_to_class(file):
    name, extension = os.path.splitext(file.name)
    if (extension == ".xlsx"):
        data = pd.read_excel(file, index_col = None, header = None).to_numpy()
    elif (extension == ".csv"):
        data = pd.read_csv(file, sep = ";", index_col = None, header = None).to_numpy()
    return Class(name, data)

@st.cache
def getData(classes):
    # check if all datasets have same length
    l = len(classes[0].data[0])
    for i in range(1, len(classes)):
        l2 = len(classes[i].data[0])
        if l != l2:
            st.write('Daten haben nicht die gleiche Länge')
            st.stop()

    x = classes[0].data
    for i in range(1, len(classes)):
        x = np.concatenate((x, classes[i].data), axis = 0)

    yList = []
    for i in range(0, len(classes)):
        for j in range(len(classes[i].data)):
            yList.append(classes[i].name)
    y = np.array(yList)

    return x, y

@st.cache
def train(x, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True)

    classifier = svm.SVC(gamma = 0.001)
    classifier.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    y_pred = classifier.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred)*100)

    classifier.fit(x, y)

    return accuracy, classifier

@st.cache
def test(uploaded_file, classifier):
    testData = file_to_class(uploaded_file).data
    pred = classifier.predict(testData)
    displayData = np.hstack((testData, np.reshape(pred, (len(pred), 1))))
    df = pd.DataFrame(data = displayData)
    df = df.set_axis([*df.columns[:-1], 'Prediction'], axis = 1, inplace = False)
    return df


class Class:
    def __init__(self, name, data):
        self.name = name
        self.data = data

st.title('Klassifizierung Messdaten')

st.write('Hier Datensätze hochladen.')
st.write('Ein Datensatz repräsentiert eine Klasse.')

uploaded_files = st.file_uploader('Klassen', accept_multiple_files = True)
classes = []
for uploaded_file in uploaded_files:
    classes.append(file_to_class(uploaded_file))

if len(classes) < 2:
    st.write('Es werden mindestens 2 Klassen benötigt.')
    st.stop()

x, y = getData(classes)

accuracy, classifier = train(x, y)

st.write('Das trainierte Modell hat eine Genauigkeit von', accuracy, '%')
st.write('Hier den Datensatz hochladen, der mit dem Modell getestet werden soll.')

uploaded_file = st.file_uploader('Testdaten')

if uploaded_file is None:
    st.stop()

df = test(uploaded_file, classifier)

st.write('Die Daten wurden klassifiziert.') # "klassifiziert?"
st.write('In der letzten Spalte befinden sich die zugeordneten Klassen.')

st.dataframe(df)
