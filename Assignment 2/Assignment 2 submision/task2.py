import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sklearn
from pandas_ml import ConfusionMatrix
from skimage.morphology import binary
from sklearn.svm import SVC
from skimage.feature import hog
from skimage import exposure
from sklearn import svm
import skimage.io
import PIL.Image as Image
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


DATADIR = 'D:/NUST/PhD/Computer Vision/Assignment 2/INRIA_Dataset_Samples_shortt/Train'
CATAGORIES = ['neg', 'pos']
########################################PROCESSING FOR TEST DATASET#################################################################################
hog_pos = []
label = []
img_arrayy =[]
def create_training_data():
    hog_image = None
    img_array = None
    for catagory in CATAGORIES:
        path = os.path.join(DATADIR, catagory)  # path to pos and neg
        print(catagory) #Added to test
        for img in os.listdir(path):
            try:
                img_array=(skimage.io.imread(os.path.join(path,img), as_gray =True))
                # im = Image.open(os.path.join(path, img))    #added here resize
                # imResize = im.resize((90, 160), Image.ANTIALIAS)    #added here resize
                # print (imResize)
                fd, hog_image = hog(img_array, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,
                            multichannel=False)
                img_arrayy.append(np.asarray(img_array))
                hog_pos.append(np.asarray(hog_image).flatten())
                label.append(catagory)
                # print ("here at hog_image")
                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                if catagory=='neg':
                    fd, hog_image_neg = hog(img_array, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                                        visualize=True, multichannel=False)
                    # print("Neg cat")
                elif catagory=='pos':
                    fd, hog_image_pos = hog(img_array, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                                        visualize=True, multichannel=False)
                    # print ("Pos cat")
                else:
                    print("Do something else")

            except Exception as e:
                pass
    return hog_pos, label

########################################PROCESSING FOR TEST DATASET#################################################################################

DATADIRTest = 'D:/NUST/PhD/Computer Vision/Assignment 2/INRIA_Dataset_Samples_shortt/Test'
CATAGORIES = ['neg', 'pos']
hog_test=[]
label_test=[]
img_arrayTestt =[]
def create_test_data():
    for catagory in CATAGORIES:
        path = os.path.join(DATADIRTest, catagory)  # path to pos and neg
        print(catagory) #Added to test
        for img in os.listdir(path):
            try:
                img_arrayTest = (skimage.io.imread(os.path.join(path,img), as_gray =True))
                # print(os.listdir(path))  #Added to test
                fd, hog_imageTest = hog(img_arrayTest, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,
                            multichannel=False)
                hog_image_rescaled = exposure.rescale_intensity(hog_imageTest, in_range=(0, 10))
                img_arrayTestt.append(np.asarray(img_arrayTest))
                hog_test.append(np.asarray(hog_imageTest).flatten())
                label_test.append(catagory)
            except Exception as e:
                pass
    return hog_test, label_test
#######################################################################################################################
getHist, getLabel= create_training_data()
get_hog_test, get_label_test = create_test_data()

clf1 = svm.SVC (gamma=0.001, C=100)
clf1 = SVC (kernel='linear')
clf1.fit (getHist, getLabel) # train , label

# clf2 = svm.SVC (gamma=0.001, C=100)
# clf2.fit (get_hog_test, get_label_test)
y_pred = clf1.predict(get_hog_test)
# print ("test labels:", get_label_test )
# print ("Histograms of test:" , get_hog_test)
# print ("predicted labels:",y_pred)
# print (get_hog_test, y_pred)

def qualitative_results():
        for i in range(10):
            plt.subplot(1,10,i+1)
            plt.imshow(img_arrayy[i], cmap='Greys_r') # train_img
            plt.axis('off')
        plt.show()
        print('Training label: %s' % (getLabel[0:10],))

        for i in range(10):
            plt.subplot(1,10,i+1)
            plt.imshow(img_arrayTestt[i], cmap='Greys_r')
            plt.axis('off')
        plt.show()
        print('Test label: %s' % (get_label_test[0:10],))


qualitative_results()


'''
##################################### ACCURACY RESULTS######################################################
# Model Accuracy: how often is the classifier correct?
print ("Accuracy of SVM")
print("Accuracy:",metrics.accuracy_score(clf1.predict(get_hog_test),get_label_test ))
# Confusion matrix for SVM
labels = ['pos', 'neg']
cm_svm = confusion_matrix(get_label_test, y_pred, labels)
########### cm_svm.classification_report
#####################################SVM Classifier######################################################
print ("Confsion matrix for SVM classifer")
print(cm_svm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm_svm)
plt.title('Confusion Matrix of the  SVM Classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print ("F1 score of SVM classifier")
print (sklearn.metrics.f1_score(get_label_test, y_pred,average='weighted'))
##################################### Random Forest Classifier######################################################

randForestclf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
randForestclf.fit(get_hog_test, get_label_test)
rf_y_pred = randForestclf.predict(get_hog_test)
# print (rf_y_pred)
print("Accuracy of Random Forest:",metrics.accuracy_score(rf_y_pred,get_label_test ))

#Confusion matrix for Random Forest Classifier
labels = ['pos', 'neg']
cm_randfor = confusion_matrix(get_label_test,rf_y_pred, labels)
print ("Confsion matrix for Random Forest classifer")
print(cm_randfor)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm_randfor)
plt.title('Confusion matrix of the Random Forest classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print ("F1 score of Random Forest classifier")
print (sklearn.metrics.f1_score(get_label_test, rf_y_pred,average='weighted')) '''