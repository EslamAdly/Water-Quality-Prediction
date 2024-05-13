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
from sklearn.svm import SVC  # Import SVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler

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
        models = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]  # Added SVM
        self.model_dropdown = tk.OptionMenu(self, self.model_var, *models)
        self.model_dropdown.pack()

        # Apply Model button
        self.apply_model_button = tk.Button(
            self, text="Apply Model", command=self.apply_model
        )
        self.apply_model_button.pack()

        # Create input text boxes
        self.create_text_boxes()

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
            ros = RandomOverSampler(random_state=109)
            X_resampled, y_resampled = ros.fit_resample(X, Y)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=109
            )
        else:
            messagebox.showerror("Error", "No data loaded!")

    def apply_model(self):
        self.split_data()
        model_name = self.model_var.get()
        if model_name == "Logistic Regression":
            self.model = self.apply_logistic_regression()
        elif model_name == "Decision Tree":
            self.model = self.apply_decision_tree()
        elif model_name == "Random Forest":
            self.model = self.apply_random_forest()
        elif model_name == "SVM":
            self.model = self.apply_svm()  # Apply SVM
        else:
            messagebox.showerror("Error", "Invalid model selection!")
            return

        if self.model:
            self.predict_and_show_results()

    def apply_logistic_regression(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        return model

    def apply_decision_tree(self):
        # Best parameters found by GridSearchCV
        best_params = {'ccp_alpha': 0.0, 'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2}
        # Create the DecisionTreeClassifier with the best parameters
        model = DecisionTreeClassifier(**best_params, random_state=109)
        # Train the model
        model.fit(self.X_train, self.y_train)
        return model

    def apply_random_forest(self):
        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        return model

    def apply_svm(self):
        # Best hyperparameters
        best_params = {'C': 1, 'gamma': 10}
        
        # Create SVC model with the best hyperparameters
        model = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
        model.fit(self.X_train, self.y_train)

        return model

    def predict_and_show_results(self):
        if self.X_test is not None and self.y_test is not None:
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            display_accuracy(accuracy)

            cm = confusion_matrix(self.y_test, y_pred)
            display_confusion_matrix(cm)
        else:
            messagebox.showerror("Error", "No data to apply model!")

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

        # Predict button
        self.predict_button = tk.Button(
            self.text_boxes_frame, text="Predict", command=self.predict
        )
        self.predict_button.grid(row=len(columns_info) + 1, column=0, sticky="w", padx=10, pady=10)

    def predict(self):
        if self.data is not None:
            input_values = [input_box.get() for input_box in self.input_boxes.values()]
            if all(input_values):
                # Convert input values to array and reshape
                input_values = np.array(input_values).reshape(1, -1)
                # Make prediction
                if self.model is not None:
                    prediction = self.model.predict(input_values)
                    messagebox.showinfo("Prediction", f"The predicted output is: {prediction}")
                else:
                    messagebox.showerror("Error", "No model selected!")
            else:
                messagebox.showerror("Error", "Please fill in all input fields!")
        else:
            messagebox.showerror("Error", "No data loaded!")

def display_accuracy(accuracy):
    accuracy_label = tk.Label(
        app, text=f"Accuracy: {accuracy:.2f}", font=("Helvetica", 12, "bold")
    )
    accuracy_label.pack()

def display_confusion_matrix(cm):
    # Convert the confusion matrix array to a string
    cm_string = "\n".join(["\t".join(map(str, row)) for row in cm])

    # Create a label to display the confusion matrix
    conf_matrix_label = tk.Label(app, text=f"Confusion Matrix:\n{cm_string}")
    conf_matrix_label.pack()



if __name__ == "__main__":
    app = ModelGUI()
    app.mainloop()
