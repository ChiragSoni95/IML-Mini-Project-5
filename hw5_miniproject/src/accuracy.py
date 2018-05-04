import numpy as np

def find_accuracies(clf,xtrain, ytrain, xtest, ytest):
    predicted1 = clf.predict ( xtrain )
    predicted2 = clf.predict ( xtest )
    trainacc = 0.0
    testacc = 0.0
    for i in range (100):
        trainaccuracy = np.mean(ytrain==predicted1)
        testaccuracy = np.mean(ytest==predicted2)
        trainacc += trainaccuracy
        testacc += testaccuracy
    return predicted1, predicted2, trainacc, testacc

#------Function to find correctly and incorrectly classified development data set-----#
def find_corincor_devdata(clf,xdev,ydev):
    predicted=clf.predict(xdev)
    correct=[]
    incorrect=[]
    for i in range ( int(len(ydev)/4) ): #Finding only few examples of correctly and incorrectly classified development data set
        if predicted[i] == ydev[i]:
            correct.append(xdev[i])
        else:
            incorrect.append(xdev[i])
    #print(correct)
    #print(incorrect)
    return correct,incorrect
