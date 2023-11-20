#imports 
import pandas as pd 
import numpy as np
import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

"""
    use of the different files : 
        Training datasets :    
            - half_catalog -> half of the catalog extracted 
            - half_classified -> same than before but cleaned/classified (for training and fine tuning)
        
            
        From the full catalog our model should read the columns and compute 
           - commitment duration, commitment duration2, Billing frequency, Billing frequency2, Consumption Model, 
           Product type 

            final output of the model should clean -> half_not_classified.csv 

            
        to evaluate accuracy :
            - full_catalog.csv 
            - full_classified.csv 


        

"""


#import datasets 
halfCat = pd.read_csv("Datasets/half_catalog.csv",delimiter=';')
fullCat = pd.read_csv("Datasets/full_catalog.csv", delimiter=';', dtype = str)
halfClassif = pd.read_csv("Datasets/half_classified.csv",delimiter=';')
halfNotClassif = pd.read_csv("Datasets/half_not_classified.csv",delimiter=';')
fullClassif = pd.read_csv("Datasets/full_classified.csv", delimiter=';')
#fullNotClassif = pd.read_csv("Datasets/full_not_classified.csv",delimiter=';')


'''
    Full catalog columns : 
        Business Unit 
        Vendor P&L 
        SKU (Stock Keeping Unit)
        Vendor Code (item)
        Vendor Code (Post)
        Description ; 2 ; 3; 4 
        


'''


#check the data 
print(fullCat.shape)
print(halfCat.shape)
print(halfNotClassif.shape)
print(halfClassif.shape)



'''
    Goal of our model : 
        clean the half_not_classified file 
        use half_classified for training 
        and full_catalog and full_classified for testing 

        
    the model we ll use : 
        Random forest 
'''




fullClassif = fullClassif.dropna()
halfClassif = halfClassif.dropna()


#try removing max
halfClassif = halfClassif.drop('Max', axis=1)
halfClassif = halfClassif.drop('ID', axis=1)
halfClassif = halfClassif.drop('SKU', axis=1)
# Extracting features and target variables for training
X_train = halfClassif  # Features in 'halfClassif'
y_train = halfClassif[['Commitment Duration', 'Commitment Duration2', 'Billing frequency', 'Billing Frequency2', 'Consumption Model', 'Product type']]  # Target columns in 'halfClassif'



# Preparing test data for prediction
'''
    drop the labels that the model is not trained on (not in halfClassif)
'''


fullClassif = fullClassif.drop('ID', axis=1)
fullClassif = fullClassif.drop('SKU', axis=1)




training_labels1 = set(X_train['Commitment Duration'].unique())

training_labels2 = set(X_train['Commitment Duration2'].unique())

mask = fullClassif['Commitment Duration'].isin(training_labels1)
fullClassif = fullClassif[mask]

mask2 = fullClassif['Commitment Duration2'].isin(training_labels2)
fullClassif = fullClassif[mask2]


training_labels1 = set(X_train['Billing frequency'].unique())

training_labels2 = set(X_train['Billing Frequency2'].unique())

mask = fullClassif['Billing frequency'].isin(training_labels1)
fullClassif = fullClassif[mask]

mask2 = fullClassif['Billing Frequency2'].isin(training_labels2)
fullClassif = fullClassif[mask2]



#get only trained on labels 
X_test = fullClassif.copy()  # Features in 'fullCat'
y_test = fullClassif.copy()


# Preprocessing categorical columns (if any)
label_encoders = {}
for col in X_train.select_dtypes(include='object').columns:
    label_encoders[col] = LabelEncoder()
    X_train[col] = label_encoders[col].fit_transform(X_train[col])

    
# Training the RandomForestClassifier model
rf_classifier = RandomForestClassifier(n_estimators=100)  
rf_classifier.fit(X_train, y_train)



# Preprocessing categorical columns in the test data (if any)
for col in X_test.select_dtypes(include='object').columns:
    X_test[col] = label_encoders[col].transform(X_test[col])  # Using the same encoders from training


# Predicting multiple columns in 'halfNotClassif'
predicted_classes = rf_classifier.predict(X_test)

print(predicted_classes[1:10])
#acc = accuracy_score(y_test, predicted_classes)

# Filling the predicted classes into 'halfNotClassif'

halfNotClassif[['Commitment Duration', 'Commitment Duration2', 'Billing frequency', 'Billing Frequency2', 'Consumption Model', 'Product type']] = predicted_classes[30582:]


halfNotClassif.to_csv('halfCleanedRf.csv', quoting=csv.QUOTE_ALL)


#metrics 

'''
    scikit learn metrics : accuracy_score, classification report ...etc
    result  : 
        ValueError: multiclass-multioutput format is not supported
'''
'''
write our own metrics (hamming loss) (designed for multiclass)
get the number of samples where the true labels differ from the predicted labels.


'''
def custom_hamming_loss(y_true, y_pred):
    # Calculate the fraction of labels that are incorrectly predicted
    incorrect_labels = sum([set(true_labels) != set(pred_labels) for true_labels, pred_labels in zip(y_true, y_pred)])
    return incorrect_labels / len(y_true)

# Calculate custom Hamming Loss
hamming_loss_value = custom_hamming_loss(X_test, predicted_classes)
print("Custom Hamming Loss:", hamming_loss_value)



