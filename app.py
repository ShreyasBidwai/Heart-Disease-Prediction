from joblib import load
import numpy as np
model=load("Trained_model")

features=['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose']
print("Enter following values:")
user_input=[]
for feature in features:
    values=float(input(f"{feature}:"))
    user_input.append(values)

user_input=np.array(user_input).reshape(1,-1)
output=model.predict(user_input)
print(output)
