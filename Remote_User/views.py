from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier
#model selection
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix, classification_report
# Create your views here.
from Remote_User.models import ClientRegister_Model,detection_type,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Threat_Status_Type(request):
    if request.method == "POST":

        if request.method == "POST":

            Flow_ID= request.POST.get('Flow_ID')
            Source_IP= request.POST.get('Source_IP')
            Source_Port= request.POST.get('Source_Port')
            Destination_IP= request.POST.get('Destination_IP')
            Destination_Port= request.POST.get('Destination_Port')
            Timestamp= request.POST.get('Timestamp')
            Flow_Duration= request.POST.get('Flow_Duration')
            Total_Fwd_Packets= request.POST.get('Total_Fwd_Packets')
            Total_Backward_Packets= request.POST.get('Total_Backward_Packets')
            Total_Length_of_Fwd_Packets= request.POST.get('Total_Length_of_Fwd_Packets')
            Total_Length_of_Bwd_Packets= request.POST.get('Total_Length_of_Bwd_Packets')
            Fwd_Packet_Length_Max= request.POST.get('Fwd_Packet_Length_Max')
            Fwd_Packet_Length_Min= request.POST.get('Fwd_Packet_Length_Min')
            Bwd_Packet_Length_Max= request.POST.get('Bwd_Packet_Length_Max')
            Flow_Bytes= request.POST.get('Flow_Bytes')
            Flow_Packets= request.POST.get('Flow_Packets')
            Fwd_Packets= request.POST.get('Fwd_Packets')
            Bwd_Packets= request.POST.get('Bwd_Packets')
            Max_Packet_Length= request.POST.get('Max_Packet_Length')



        data = pd.read_csv("Network_Datasets.csv", encoding='latin-1')


        def apply_results(label):
            if (label == "Benign"):
                return 0  # No Threat
            elif (label == "Threat"):
                return 1  # Threat

        data['Label'] = data['Class'].apply(apply_results)

        x = data['Flow_ID'].apply(str)
        y = data['Label']

        cv = CountVectorizer()

        print(x)
        print(y)

        x = cv.fit_transform(x)


        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print(naivebayes)
        print(confusion_matrix(y_test, predict_nb))
        print(classification_report(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        Flow_ID1 = [Flow_ID]
        vector1 = cv.transform(Flow_ID1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'Benign'
        elif prediction == 1:
            val = 'Threat'

        print(prediction)
        print(val)

        detection_type.objects.create(
        Flow_ID=Flow_ID,
        Source_IP=Source_IP,
        Source_Port=Source_Port,
        Destination_IP=Destination_IP,
        Destination_Port=Destination_Port,
        Timestamp=Timestamp,
        Flow_Duration=Flow_Duration,
        Total_Fwd_Packets=Total_Fwd_Packets,
        Total_Backward_Packets=Total_Backward_Packets,
        Total_Length_of_Fwd_Packets=Total_Length_of_Fwd_Packets,
        Total_Length_of_Bwd_Packets=Total_Length_of_Bwd_Packets,
        Fwd_Packet_Length_Max=Fwd_Packet_Length_Max,
        Fwd_Packet_Length_Min=Fwd_Packet_Length_Min,
        Bwd_Packet_Length_Max=Bwd_Packet_Length_Max,
        Flow_Bytes=Flow_Bytes,
        Flow_Packets=Flow_Packets,
        Fwd_Packets=Fwd_Packets,
        Bwd_Packets=Bwd_Packets,
        Max_Packet_Length=Max_Packet_Length,
        Prediction=val)

        return render(request, 'RUser/Predict_Threat_Status_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Threat_Status_Type.html')



