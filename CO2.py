import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
import matplotlib.pyplot as plt


df = pd.read_csv('CO2 Emissions_Canada (1).csv')


df = df.dropna()


numerical_columns = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 
                     'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 
                     'Fuel Consumption Comb (mpg)']

X = df[numerical_columns]  
y = df["CO2 Emissions(g/km)"]  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = linear_model.LinearRegression()


model.fit(X_train, y_train)


root = tk.Tk()
root.title("CO2 Emission Prediction")
root.geometry("600x700")
root.config(bg="#f0f0f0")


label = tk.Label(root, text="Please enter the car's features", font=("Arial", 14), bg="#f0f0f0")
label.pack(pady=20)

engine_size_label = tk.Label(root, text="Engine Size (L):", font=("Arial", 12), bg="#f0f0f0")
engine_size_label.pack(pady=5)
engine_size_entry = tk.Entry(root, font=("Arial", 12))
engine_size_entry.pack(pady=5)

cylinders_label = tk.Label(root, text="Number of Cylinders:", font=("Arial", 12), bg="#f0f0f0")
cylinders_label.pack(pady=5)
cylinders_entry = tk.Entry(root, font=("Arial", 12))
cylinders_entry.pack(pady=5)

fuel_consumption_city_label = tk.Label(root, text="Fuel Consumption City (L/100 km):", font=("Arial", 12), bg="#f0f0f0")
fuel_consumption_city_label.pack(pady=5)
fuel_consumption_city_entry = tk.Entry(root, font=("Arial", 12))
fuel_consumption_city_entry.pack(pady=5)

fuel_consumption_hwy_label = tk.Label(root, text="Fuel Consumption Hwy (L/100 km):", font=("Arial", 12), bg="#f0f0f0")
fuel_consumption_hwy_label.pack(pady=5)
fuel_consumption_hwy_entry = tk.Entry(root, font=("Arial", 12))
fuel_consumption_hwy_entry.pack(pady=5)

fuel_consumption_comb_label = tk.Label(root, text="Fuel Consumption Comb (L/100 km):", font=("Arial", 12), bg="#f0f0f0")
fuel_consumption_comb_label.pack(pady=5)
fuel_consumption_comb_entry = tk.Entry(root, font=("Arial", 12))
fuel_consumption_comb_entry.pack(pady=5)

fuel_consumption_comb_mpg_label = tk.Label(root, text="Fuel Consumption Comb (mpg):", font=("Arial", 12), bg="#f0f0f0")
fuel_consumption_comb_mpg_label.pack(pady=5)
fuel_consumption_comb_mpg_entry = tk.Entry(root, font=("Arial", 12))
fuel_consumption_comb_mpg_entry.pack(pady=5)


history = []


def predict_co2():
    try:
        
        engine_size = float(engine_size_entry.get())
        cylinders = int(cylinders_entry.get())
        fuel_consumption_city = float(fuel_consumption_city_entry.get())
        fuel_consumption_hwy = float(fuel_consumption_hwy_entry.get())
        fuel_consumption_comb = float(fuel_consumption_comb_entry.get())
        fuel_consumption_comb_mpg = float(fuel_consumption_comb_mpg_entry.get())
        
        
        input_data = pd.DataFrame({
            'Engine Size(L)': [engine_size],
            'Cylinders': [cylinders],
            'Fuel Consumption City (L/100 km)': [fuel_consumption_city],
            'Fuel Consumption Hwy (L/100 km)': [fuel_consumption_hwy],
            'Fuel Consumption Comb (L/100 km)': [fuel_consumption_comb],
            'Fuel Consumption Comb (mpg)': [fuel_consumption_comb_mpg]
        })
        
        predicted_co2 = model.predict(input_data)

        result_label.config(text=f"Predicted CO2 Emission: {predicted_co2[0]:.2f} g/km", fg="green")
        
        history.append(f"Engine Size: {engine_size}, Cylinders: {cylinders}, City Fuel Consumption: {fuel_consumption_city}, Hwy Fuel Consumption: {fuel_consumption_hwy}, Comb Fuel Consumption: {fuel_consumption_comb}, Predicted CO2: {predicted_co2[0]:.2f} g/km")
        
    except ValueError:
        messagebox.showerror("Error", "Please enter all fields correctly.")

def clear_inputs():
    engine_size_entry.delete(0, tk.END)
    cylinders_entry.delete(0, tk.END)
    fuel_consumption_city_entry.delete(0, tk.END)
    fuel_consumption_hwy_entry.delete(0, tk.END)
    fuel_consumption_comb_entry.delete(0, tk.END)
    fuel_consumption_comb_mpg_entry.delete(0, tk.END)


def clear_history():
    history_listbox.delete(0, tk.END)


predict_button = tk.Button(root, text="Predict CO2 Emission", font=("Arial", 14), bg="#4CAF50", fg="white", command=predict_co2)
predict_button.pack(pady=20)


clear_button = tk.Button(root, text="Clear Inputs", font=("Arial", 14), bg="#f44336", fg="white", command=clear_inputs)
clear_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14), bg="#f0f0f0")
result_label.pack(pady=10)

history_label = tk.Label(root, text="History", font=("Arial", 14), bg="#f0f0f0")
history_label.pack(pady=10)

history_listbox = tk.Listbox(root, width=50, height=10, font=("Arial", 10))
history_listbox.pack(pady=10)

clear_history_button = tk.Button(root, text="Clear History", font=("Arial", 14), bg="#f44336", fg="white", command=clear_history)
clear_history_button.pack(pady=10)

root.mainloop()

