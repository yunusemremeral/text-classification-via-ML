import numpy as np
from tkinter import filedialog
from tkinter import *
from sklearn.datasets import load_files
import tecnique.Bagging as Bagging
import tecnique.Boost as Boosting
import tecnique.Rocchio_classification as RocchioClassification
import tecnique.MultinomialNB as NaiveBayesClassifier
import tecnique.K_nearest_Neighbor as KNearestNeighbor
import tecnique.SVM as SupportVectorMachine
import tecnique.Random_Forest as RandomForest
import tecnique.Decision_Tree as DecisionTree
import tecnique.RNN as RNN
import tecnique.DNN as DNN
import tecnique.CNN as CNN


root = Tk()
root.title('Text Classification')
root.geometry("1200x750")
root.resizable(0, 0)

#configure the grid
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)

labelDir=Label(root, text="Text Classification Tecnique :")
labelDir.grid(column=0,row=0, sticky=E, padx=5, pady=5)
labelDir=Label(root, text="Dataset folder :")
labelDir.grid(column=0,row=1, sticky=E, padx=5, pady=5)
labelDir=Label(root, text="Categories :")
labelDir.grid(column=0,row=2, sticky=E, padx=5, pady=5)

clicked = StringVar()
clicked.set("Rocchio classification")
drop = OptionMenu(root, clicked, "Rocchio classification",
"Boosting",
"Bagging",
"Naive Bayes Classifier",
"K-nearest Neighbor",
"Support Vector Machine (SVM)",
"Decision Tree",
"Random Forest"
)
drop.grid(column=1,row=0,sticky=W, padx=5, pady=5)
 



def loadDataSet():
    global dataset      
    root.directory = filedialog.askdirectory(initialdir="", title="select Dataset")
    dataset = load_files(root.directory, encoding="utf-8", decode_error="replace")
    labels, counts = np.unique(dataset.target, return_counts=True)
    labels_str = np.array(dataset.target_names)[labels]
    labelDir=Label(root, text=dict(zip(labels_str, counts)))
    labelDir.grid(column=1,row=2, sticky=W, padx=5, pady=5)



def ggwp():
    tech = clicked.get()
    if tech == "Boosting":
        result = Boosting.run(dataset)
    elif tech == "Bagging":
        result = Bagging.run(dataset)
    elif tech == "Rocchio classification":
        result = RocchioClassification.run(dataset)
    elif tech == "Naive Bayes Classifier":
        result = NaiveBayesClassifier.run(dataset)
    elif tech == "K-nearest Neighbor":
        result = KNearestNeighbor.run(dataset)
    elif tech == "Support Vector Machine (SVM)":
        result = SupportVectorMachine.run(dataset)
    elif tech == "Decision Tree":
        result = DecisionTree.run(dataset)
    elif tech == "Random Forest":
        result = RandomForest.run(dataset)
    elif tech == "RNN":
        result = RNN.run(dataset)
    elif tech == "DNN":
        result = DNN.run(dataset)
    elif tech == "CNN":
        result = CNN.run(dataset)

    text_box = Text(root,height=12,width=100)
    text_box.grid(column=1, row=4, sticky=W, padx=5, pady=5)
    text_box.insert('end', (tech + " - Result\n\n"))
    text_box.insert('end', result)
    

importBtn = Button(root, text="Select Folder", command=loadDataSet)
importBtn.grid(column=1,row=1, sticky=W, padx=5, pady=5)
runBtn = Button(root, text="Run Run!", command=ggwp)
runBtn.grid(column=1, row=3, sticky=W, padx=5, pady=5)

root.mainloop()