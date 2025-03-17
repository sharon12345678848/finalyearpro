from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import socket
from threading import Thread 

main = tkinter.Tk()
main.title("Machine Learning based Health Prediction System using IBM Cloud as PaaS")
main.geometry("1300x1200")


global filename
global knn_precision,nb_precision,tree_precision,svm_precision,random_precision,logistic_precision,gb_precision,mlp_precision,ensemble_precision
global knn_recall,nb_recall,tree_recall,svm_recall,random_recall,logistic_recall,gb_recall,mlp_recall,ensemble_recall
global knn_fmeasure,nb_fmeasure,tree_fmeasure,svm_fmeasure,random_fmeasure,logistic_fmeasure,gb_fmeasure,mlp_fmeasure,ensemble_fmeasure
global knn_acc,nb_acc,tree_acc,svm_acc,random_acc,logistic_acc,gb_acc,mlp_acc,ensemble_acc

global classifier
global X_train, X_test, y_train, y_test

def upload():
    global filename
    global X_train, X_test, y_train, y_test
    filename = filedialog.askopenfilename(initialdir = "dataset")
    pathlabel.config(text=filename)
    dataset = pd.read_csv(filename)
    cols = dataset.shape[1]
    cols = cols - 1
    X = dataset.values[:, 0:cols] 
    Y = dataset.values[:, cols]
    Y = Y.astype('int')
   
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
     
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded')
    text.insert(END,"Total dataset size : "+str(len(dataset)))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,'Healthcare Training & Testing data generated\n\n')
    text.insert(END,"Total Splitted training size : "+str(len(X_train))+"\n")
    text.insert(END,"Total Splitted testing size : "+str(len(X_test)))

    

def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
def KNN():
    global knn_precision
    global knn_recall
    global knn_fmeasure
    global knn_acc
    text.delete('1.0', END)
    cls = KNeighborsClassifier(n_neighbors = 10) 
    cls.fit(X_train, y_train) 
    text.insert(END,"KNN Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls)
    knn_precision = precision_score(y_test, prediction_data,average='macro') * 100
    knn_recall = recall_score(y_test, prediction_data,average='macro') * 100
    knn_fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    knn_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"KNN Precision : "+str(knn_precision)+"\n")
    text.insert(END,"KNN Recall : "+str(knn_recall)+"\n")
    text.insert(END,"KNN FMeasure : "+str(knn_fmeasure)+"\n")
    text.insert(END,"KNN Accuracy : "+str(knn_acc)+"\n")
    #classifier = cls
    
def naivebayes():
    global nb_precision
    global nb_recall
    global nb_fmeasure
    global nb_acc
    text.delete('1.0', END)
    cls = BernoulliNB(binarize=0.0)
    cls.fit(X_train, y_train)
    text.insert(END,"Naive Bayes Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    nb_precision = precision_score(y_test, prediction_data,average='macro') * 100
    nb_recall = recall_score(y_test, prediction_data,average='macro') * 100
    nb_fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    nb_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Naive Bayes Precision : "+str(nb_precision)+"\n")
    text.insert(END,"Naive Bayes Recall : "+str(nb_recall)+"\n")
    text.insert(END,"Naive Bayes FMeasure : "+str(nb_fmeasure)+"\n")
    text.insert(END,"Naive Bayes Accuracy : "+str(nb_acc)+"\n")
    
    

def decisionTree():
    text.delete('1.0', END)
    global tree_acc
    global tree_precision
    global tree_recall
    global tree_fmeasure
    rfc = DecisionTreeClassifier(criterion = "entropy", splitter = "random", max_depth = 20,  min_samples_split = 50, min_samples_leaf = 20, max_features = 5)
    rfc.fit(X_train, y_train)
    text.insert(END,"Decision Tree Prediction Results\n") 
    prediction_data = prediction(X_test, rfc) 
    tree_precision = precision_score(y_test, prediction_data,average='macro') * 100
    tree_recall = recall_score(y_test, prediction_data,average='macro') * 100
    tree_fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    tree_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Decision Tree Precision : "+str(tree_precision)+"\n")
    text.insert(END,"Decision Tree Recall : "+str(tree_recall)+"\n")
    text.insert(END,"Decision Tree FMeasure : "+str(tree_fmeasure)+"\n")
    text.insert(END,"Decision Tree Accuracy : "+str(tree_acc)+"\n")
    
def randomForest():
    text.delete('1.0', END)
    global random_acc
    global random_precision
    global random_recall
    global random_fmeasure
    rfc = RandomForestClassifier(n_estimators=200, random_state=0)
    rfc.fit(X_train, y_train)
    text.insert(END,"Random Forest Prediction Results\n") 
    prediction_data = prediction(X_test, rfc) 
    random_precision = precision_score(y_test, prediction_data,average='macro') * 100
    random_recall = recall_score(y_test, prediction_data,average='macro') * 100
    random_fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    random_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Random Forest Precision : "+str(random_precision)+"\n")
    text.insert(END,"Random Forest Recall : "+str(random_recall)+"\n")
    text.insert(END,"Random Forest FMeasure : "+str(random_fmeasure)+"\n")
    text.insert(END,"Random Forest Accuracy : "+str(random_acc)+"\n")

def logisticRegression():
    text.delete('1.0', END)
    global logistic_acc
    global logistic_precision
    global logistic_recall
    global logistic_fmeasure
    rfc = LogisticRegression(penalty='l2', tol=0.002, C=2.0)
    rfc.fit(X_train, y_train)
    text.insert(END,"Logistic Regression Prediction Results\n") 
    prediction_data = prediction(X_test, rfc) 
    logistic_precision = precision_score(y_test, prediction_data,average='macro') * 100
    logistic_recall = recall_score(y_test, prediction_data,average='macro') * 100
    logistic_fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    logistic_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Logistic Regression Precision : "+str(logistic_precision)+"\n")
    text.insert(END,"Logistic Regression Recall : "+str(logistic_recall)+"\n")
    text.insert(END,"Logistic Regression FMeasure : "+str(logistic_fmeasure)+"\n")
    text.insert(END,"Logistic Regression Accuracy : "+str(logistic_acc)+"\n")
    
def SVM():
    text.delete('1.0', END)
    global svm_acc
    global svm_precision
    global svm_recall
    global svm_fmeasure
    global X_train, X_test, y_train, y_test
    rfc = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2)
    rfc.fit(X_train, y_train)
    text.insert(END,"SVM Prediction Results\n") 
    prediction_data = prediction(X_test, rfc) 
    svm_precision = precision_score(y_test, prediction_data,average='macro') * 100
    svm_recall = recall_score(y_test, prediction_data,average='macro') * 100
    svm_fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    svm_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"SVM Precision : "+str(svm_precision)+"\n")
    text.insert(END,"SVM Recall : "+str(svm_recall)+"\n")
    text.insert(END,"SVM FMeasure : "+str(svm_fmeasure)+"\n")
    text.insert(END,"SVM Accuracy : "+str(svm_acc)+"\n")

    
    
def graidentBoosting():
    global gb_acc
    global gb_precision
    global gb_recall
    global gb_fmeasure
    text.delete('1.0', END)
    cls = GradientBoostingClassifier(random_state=0)
    cls.fit(X_train, y_train)
    text.insert(END,"Gradient Boosting Classifier Prediction Results\n") 
    prediction_data = prediction(X_test, cls) 
    gb_precision = precision_score(y_test, prediction_data,average='macro') * 100
    gb_recall = recall_score(y_test, prediction_data,average='macro') * 100
    gb_fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    gb_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Gradient Boosting Precision : "+str(gb_precision)+"\n")
    text.insert(END,"Gradient Boosting Recall : "+str(gb_recall)+"\n")
    text.insert(END,"Gradient Boosting FMeasure : "+str(gb_fmeasure)+"\n")
    text.insert(END,"Gradient Boosting Accuracy : "+str(gb_acc)+"\n")
    

def MLPAlgorithm():
    global mlp_acc
    global mlp_precision
    global mlp_recall
    global mlp_fmeasure
    text.delete('1.0', END)
    cls = MLPClassifier(random_state=1, max_iter=10)
    cls.fit(X_train, y_train)
    text.insert(END,"Multilayer Perceptron Classifier Prediction Results\n") 
    prediction_data = prediction(X_test, cls) 
    mlp_precision = precision_score(y_test, prediction_data,average='macro') * 100
    mlp_recall = recall_score(y_test, prediction_data,average='macro') * 100
    mlp_fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    mlp_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Multilayer Perceptron Precision : "+str(mlp_precision)+"\n")
    text.insert(END,"Multilayer Perceptron Recall : "+str(mlp_recall)+"\n")
    text.insert(END,"Multilayer Perceptron FMeasure : "+str(mlp_fmeasure)+"\n")
    text.insert(END,"Multilayer Perceptron Accuracy : "+str(mlp_acc)+"\n")

def ensembleAlgorithm():
    global ensemble_acc
    global ensemble_precision
    global ensemble_recall
    global ensemble_fmeasure
    global classifier
    text.delete('1.0', END)

    svc_alg = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2)
    rf = RandomForestClassifier(n_estimators=200, random_state=0)
    dt = DecisionTreeClassifier(criterion = "entropy", splitter = "random", max_depth = 20,  min_samples_split = 50, min_samples_leaf = 20, max_features = 5)
    
    estimators = []
    estimators.append(('tree', dt))
    estimators.append(('svm', svc_alg))
    estimators.append(('rf', rf))
    ensemble = VotingClassifier(estimators)
    ensemble.fit(X_train, y_train)
    text.insert(END,"Ensemble Classifier Prediction Results\n") 
    prediction_data = prediction(X_test, ensemble)
    classifier = ensemble
    ensemble_precision = precision_score(y_test, prediction_data,average='macro') * 100
    ensemble_recall = recall_score(y_test, prediction_data,average='macro') * 100
    ensemble_fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    ensemble_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Ensemble  Precision : "+str(ensemble_precision)+"\n")
    text.insert(END,"Ensemble  Recall   : "+str(ensemble_recall)+"\n")
    text.insert(END,"Ensemble  FMeasure : "+str(ensemble_fmeasure)+"\n")
    text.insert(END,"Ensemble  Accuracy : "+str(ensemble_acc)+"\n")


def startApplicationServer():
    class ClientThread(Thread): 
 
        def __init__(self,ip,port): 
            Thread.__init__(self) 
            self.ip = ip 
            self.port = port 
            text.insert(END,'Request received from Client IP : '+ip+' with port no : '+str(port)+"\n") 
   
        def run(self):
            headers = 'age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal'
            data = conn.recv(1024).decode()
            f = open("test.txt", "w")
            f.write(headers+"\n"+str(data))
            f.close()
            text.insert(END,"from connected user: " + str(data)+"\n")
            test = pd.read_csv('test.txt')
            predict = classifier.predict(test)
            print(predict[0])
            msg = ''
            if predict[0] == 0:
                msg = 'Prediction Result : Patient condition is stable detected'
            else:
                msg = 'Prediction Result : Patient condition is abnormal detected'
            data = str(msg)
            text.insert(END,data+"\n")
            main.update_idletasks()
            conn.send(data.encode())


    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    server.bind(('localhost', 5000))
    threads = []
    text.insert(END,"Server Started & waiting for incoming connections\n\n")
    running = True
    while running:
        server.listen(4)
        (conn, (ip,port)) = server.accept()
        newthread = ClientThread(ip,port) 
        newthread.start() 
        threads.append(newthread) 
    for t in threads:
        t.join()

def startServer():
    Thread(target=startApplicationServer).start()

def precisionGraph():
    height = [knn_precision,nb_precision,tree_precision,svm_precision,random_precision,gb_precision,mlp_precision,ensemble_precision]
    bars = ('KNN Precision', 'NB Precision','DT Precision','SVM Precision','RF Precision','Gradient Boosting Precision','MLP Precision','Ensemble Precision')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

    
def recallGraph():
    height = [knn_recall,nb_recall,tree_recall,svm_recall,random_recall,gb_recall,mlp_recall,ensemble_recall]
    bars = ('KNN Recall', 'NB Recall','DT Recall','SVM Recall','RF Recall','Gradient Boosting Recall','MLP Recall','Ensemble Recall')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    
def fscoreGraph():
    height = [knn_fmeasure,nb_fmeasure,tree_fmeasure,svm_fmeasure,random_fmeasure,gb_fmeasure,mlp_fmeasure,ensemble_fmeasure]
    bars = ('KNN FScore', 'NB FScore','DT FScore','SVM FScore','RF FScore','Gradient Boosting  FScore','MLP FScore','Ensemble FScore')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()  
    
def accuracyGraph():
    height = [knn_acc,nb_acc,tree_acc,svm_acc,random_acc,gb_acc,mlp_acc,ensemble_acc]
    bars = ('KNN ACC', 'NB ACC','DT ACC','SVM ACC','RF ACC','Gradient Boosting ACC','MLP ACC','Ensemble Acc')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()  

font = ('times', 16, 'bold')
title = Label(main, text='Machine Learning based Health Prediction System using IBM Cloud as PaaS')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Healthcare Dataset & Preprocess", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

svmButton = Button(main, text="Run SVM Algorithm", command=SVM)
svmButton.place(x=700,y=200)
svmButton.config(font=font1) 

knnButton = Button(main, text="Run KNN Algorithm", command=KNN)
knnButton.place(x=700,y=250)
knnButton.config(font=font1) 

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=naivebayes)
nbButton.place(x=700,y=300)
nbButton.config(font=font1)

treeButton = Button(main, text="Run Decision Tree Algorithm", command=decisionTree)
treeButton.place(x=700,y=350)
treeButton.config(font=font1)

#lrButton = Button(main, text="Run Logistic Regression Algorithm", command=logisticRegression)
#lrButton.place(x=700,y=400)
#lrButton.config(font=font1)


randomButton = Button(main, text="Run Random Forest Algorithm", command=randomForest)
randomButton.place(x=700,y=450)
randomButton.config(font=font1)

mlpButton = Button(main, text="Run MLP Algorithm", command=MLPAlgorithm)
mlpButton.place(x=1000,y=450)
mlpButton.config(font=font1)

gbButton = Button(main, text="Run Gradient Boosting Algorithm", command=graidentBoosting)
gbButton.place(x=700,y=500)
gbButton.config(font=font1)

mlpButton = Button(main, text="Run Ensemble Algorithm", command=ensembleAlgorithm)
mlpButton.place(x=1000,y=500)
mlpButton.config(font=font1)

graphButton = Button(main, text="Precision Graph", command=precisionGraph)
graphButton.place(x=700,y=550)
graphButton.config(font=font1)

recallButton = Button(main, text="Recall Graph", command=recallGraph)
recallButton.place(x=900,y=550)
recallButton.config(font=font1)

scoreButton = Button(main, text="Fscore Graph", command=fscoreGraph)
scoreButton.place(x=700,y=600)
scoreButton.config(font=font1)

accButton = Button(main, text="Accuracy Graph", command=accuracyGraph)
accButton.place(x=900,y=600)
accButton.config(font=font1)

predictButton = Button(main, text="Start Cloud Server", command=startServer)
predictButton.place(x=700,y=650)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
