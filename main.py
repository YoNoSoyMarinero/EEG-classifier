from RawData import RawData
from FeatureExtraction import FeatureExtraction
from EMGDataFrame import EMGDataFrame
from ClassificationEMG import ClassificationEMG
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

        

class App(tk.Tk):

    def load_data(self):
        try:
            data = RawData(test_name='emg_test.mat', train_name='emg_train.mat')
            train_data_np = FeatureExtraction.featutre_extraction(data.emg_train_chs, data.emg_train_type)
            test_data_np = FeatureExtraction.featutre_extraction(data.emg_test_chs, data.emg_test_type)
            self.df = EMGDataFrame(train_data_np, test_data_np)
            messagebox.showinfo("Info!", "Data loaded successfully!")
        except:
            messagebox.showerror("Error", "Failed to load data!")

    def train_model(self):
        try:
            self.model = ClassificationEMG().classification_model_train(self.df, self.combo_box.get())
            messagebox.showinfo("Info!", "Model trained successfully!")
        except:
            messagebox.showerror("Error", "Failed to train the model!")
    
    def test_model(self):
        self.cm, self.accuracy = ClassificationEMG.classification_model_test(self.df, self.model)
        self.accuracy_str_var.set(str(self.accuracy * 100) + "%")
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, display_labels=['1', '2', '3', '4', '5', '6'])
        disp.plot(cmap="YlOrBr_r")
        plt.show()

    def __init__(self) -> None:
        super().__init__()
        self.title("App")
        self.background_color = "gray17"
        self.font_color = "black"
        self.geometry("1200x800")
        self.config(bg=self.background_color)
        self.df = None
        self.model = None
        self.cm = None
        self.accuracy = None

        self.accuracy_str_var = tk.StringVar()

        self.load_button = tk.Button(self, text="LOAD", bg=self.background_color, fg="lime green", font=30, width=17, height=4, command=self.load_data)
        self.load_button.grid(row=0, column=0, pady=40, padx=50)

        self.train_button = tk.Button(self, text="TRAIN", bg=self.background_color, fg="IndianRed1", font=30, width=17, height=4, command=self.train_model)
        self.train_button.grid(row=1, column=0, pady=50, padx=50)

        self.test_button = tk.Button(self, text="TEST", bg=self.background_color, fg="cyan3", font=30, width=17, height=4, command=self.test_model)
        self.test_button.grid(row=2, column=0, pady=60, padx=50)

        tk.Label(self, text="Pick model: ", bg=self.background_color, fg="gray83", font=20, width=17, height=4).grid(row= 0, column = 1, pady = 0, padx=0)
        self.combo_box = ttk.Combobox(self, value=['linear', 'quadratic', 'knn', 'random_forest'], background=self.background_color, foreground=self.font_color, font=30, width=17, height=4)
        self.combo_box.current(0)
        self.combo_box.grid(row = 0, column = 2, pady = 0, padx=0)

        tk.Label(self, text="Accuracy: ", bg=self.background_color, fg="gray83", font=20, width=17, height=4).grid(row= 1, column = 1, pady = 0, padx=0)
        self.accuracy_label = tk.Label(self, fg=self.font_color, bg=self.background_color, font=20, textvariable=self.accuracy_str_var)
        self.accuracy_label.grid(row = 1, column = 2, pady = 0, padx=0)
       
        




def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
