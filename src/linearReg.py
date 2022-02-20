import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
import cython

# input = ["(0.48 0.70)", "(0.13 0.66)", "(0.52 0.66)", "(0.48 0.20)", "(0.18 0.66)", "(0.14 0.80)", "(0.20 0.66)", "(0.48 0.66)", "(0.50 0.66)", "(0.13 0.66)", "(0.60 0.66)", "(0.12 0.66)", "(0.11 0.66)"]
def Accuracy(input):
    leftHandx = str(input[4])[2:6]
    leftFootx = str(input[10])[2:6]
    #rightHandy = str(input[7])[7:10]
    #x1 =  str(df[5][0])[2:6]
    #y1 =  str(df[5][0])[7:11]
    #input = 0.44
    print("left hand x: ", leftHandx)
    print("left foot x: ", leftFootx)
    diff = abs(float(leftHandx)-float(leftFootx))
    if (float(leftFootx) == 0):
        return 0
    accuracy = 1 - (diff/float(leftFootx))
    # print("accuracy: ", accuracy)
    return accuracy
    #accuracy = abs(input/float(y1))
    #print("accuracy:", accuracy)

"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)
model = LogisticRegression(max_depth= 2)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(accuracy_score(pred, y_test))
"""
