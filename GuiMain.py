import tkinter as tk
import numpy as np
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


class ModelGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Model GUI")
        self.geometry("800x600")

        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.text_boxes_frame = None
        # Load Data button

        self.load_data_button = tk.Button(
            self, text="Load Data", command=self.load_data
        )
        self.load_data_button.pack()
        # Model selection
        self.model_var = tk.StringVar()
        self.model_var.set("Logistic Regression")
        models = ["Logistic Regression", "Decision Tree", "Random Forest"]
        self.model_dropdown = tk.OptionMenu(self, self.model_var, *models)
        self.model_dropdown.pack()

        # Apply Model button
        self.apply_model_button = tk.Button(
            self, text="Apply Model", command=self.apply_model
        )
        self.apply_model_button.pack()

    def load_data(self):
        file_path = "processed_data.xlsx"
        if file_path:
            try:
                self.data = pd.read_excel(file_path)
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading data: {str(e)}")

    def split_data(self):
        if self.data is not None:
            X = self.data.drop(columns=["Potability"])
            Y = self.data["Potability"]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, Y, test_size=0.2
            )
        else:
            messagebox.showerror("Error", "No data loaded!")

    # to validate input
    def is_valid_number(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    # ===================

    def apply_model(self):
        self.train_model()
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        if self.X_train is not None and self.y_train is not None:
            messagebox.showinfo("Model Results", f"Accuracy: {accuracy:.2f}")
            # confusion_matrix
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")
            plt.show()

            if self.text_boxes_frame is None:
                self.create_text_boxes()
        else:
            messagebox.showerror("Error", "No data to apply model!")

    def train_model(self):
        self.split_data()
        Scaler = StandardScaler()
        X_train_scaled = Scaler.fit_transform(self.X_train)
        self.X_test = Scaler.transform(self.X_test)
        if self.X_train is not None and self.y_train is not None:
            model_name = self.model_var.get()
            if model_name == "Logistic Regression":
                self.model = LogisticRegression()
            elif model_name == "Decision Tree":
                self.model = DecisionTreeClassifier()
            elif model_name == "Random Forest":
                self.model = RandomForestClassifier()
            else:
                messagebox.showerror("Error", "Invalid model selection!")
                return
        else:
            messagebox.showerror("Error", "No data to apply model!")
        self.model.fit(X_train_scaled, self.y_train)

    # ================================================================================================================
    def create_text_boxes(self):
        self.text_boxes_frame = ttk.Frame(self)
        self.text_boxes_frame.pack(fill=tk.BOTH, expand=True)
        columns_info = [
            "ph",
            "Hardness",
            "Solids",
            "Chloramines",
            "Sulfate",
            "Conductivity",
            "Organic_carbon",
            "Trihalomethanes",
            "Turbidity",
        ]
        self.input_boxes = {}

        # input fields
        for i, column in enumerate(columns_info):
            # name   ,library    ,parent                 ,text
            label = ttk.Label(self.text_boxes_frame, text=f"{i}   {column}")
            label.grid(row=i, column=0, sticky="w", padx=10, pady=5)
            input_box = ttk.Entry(self.text_boxes_frame, width=20)
            input_box.grid(row=i, column=1, sticky="w", padx=10, pady=5)
            self.input_boxes[column] = input_box

        # predict output
        protLabel = ttk.Label(self.text_boxes_frame, text=f"Potability")
        protLabel.grid(row=15, column=0, sticky="w", padx=20, pady=45)
        self.answer = ttk.Label(self.text_boxes_frame, text=f"")
        self.answer.grid(row=15, column=1, sticky="w", padx=20, pady=45)
        self.waterState = ttk.Label(self.text_boxes_frame, text=f"")
        self.waterState.grid(row=17, column=0, sticky="w", padx=20, pady=10)
        self.waterState.configure(font="sans", width=20)
        # Button to get user input
        self.predict_button = ttk.Button(
            self, text="Predict", command=self.get_user_input
        )
        self.predict_button.pack()

    # ========================================================================================================
    def get_user_input(self):
        self.train_model()
        user_input = []
        for column, input_box in self.input_boxes.items():
            value = input_box.get()
            if self.is_valid_number(value):
                user_input.append(float(value))
            else:
                messagebox.showerror("Error", f"Invalid input for {column}: {value}")
                return
        arr = np.array(user_input)

        # reshape ->to reshape the array into a format can used as input for prediction()
        res = self.model.predict(arr.reshape(1, -1))

        print(res)
        self.answer["text"] = str(res)
        if res == 1:
            self.waterState["text"] = "water safe to drink"
        else:
            self.waterState["text"] = "water is not safe to drink"


if __name__ == "__main__":
    app = ModelGUI()
    app.mainloop()
