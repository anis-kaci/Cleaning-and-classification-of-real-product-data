#imports 
import pandas as pd 
import numpy as np
import csv

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


print(fullCat.shape)
print(halfCat.shape)
print(halfNotClassif.shape)
print(halfClassif.shape)


'''
    Step 1 : we must compute from the full catalog : 
        - Commitment duration 
        - Commitment Duration2 
        - Billing frequency 
        - Billing frequency2
        - Consumption Model 
        - Product type 

'''



'''
print(fullCat.columns)

prod_type_df = fullCat[['SKU', 'Product Type']]

print(prod_type_df.head())

#halfclassif to test 
testHalfClassif = pd.read_csv('Datasets/half_not_classified.csv',delimiter=';')

print(testHalfClassif.head())

#test the merge (classify product type)
testHalfClassif = pd.merge(halfNotClassif, prod_type_df, on='SKU', how='left')

testHalfClassif.drop('Product type',axis=1, inplace=True)

print(testHalfClassif.sample(6))
'''


'''
    Goal of our model : 
        clean the half_not_classified file 
        use half_classified for training 
        and full_catalog and full_classified for testing 

        
    the model we ll use : 
        we will try random forest 
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Assuming 'halfClassif', 'halfNotClassif', and 'fullCat' are your DataFrames
# Assuming multiple columns need to be predicted in 'halfNotClassif'

fullClassif = fullClassif.dropna()
halfClassif = halfClassif.dropna()


#try removing max
halfClassif = halfClassif.drop('Max', axis=1)
halfClassif = halfClassif.drop('ID', axis=1)
halfClassif = halfClassif.drop('SKU', axis=1)
# Extracting features and target variables for training
X_train = halfClassif  # Features in 'halfClassif'
y_train = halfClassif[['Commitment Duration', 'Commitment Duration2', 'Billing frequency', 'Billing Frequency2', 'Consumption Model', 'Product type']]  # Target columns in 'halfClassif'

'''
#test 
y_train = y_train.drop('Commitment Duration', axis=1)
y_train = y_train.drop('Commitment Duration2', axis=1)

X_train = X_train.drop('Commitment Duration', axis=1)
X_train = X_train.drop('Commitment Duration2', axis=1)
'''


# Preparing test data for prediction
'''
    drop the labels that the model is not trained on (not in halfClassif)
'''


fullClassif = fullClassif.drop('ID', axis=1)
fullClassif = fullClassif.drop('SKU', axis=1)

'''
#test 
fullClassif = fullClassif.drop('Commitment Duration', axis=1)
fullClassif = fullClassif.drop('Commitment Duration2', axis=1)

mask = fullClassif['Billing frequency'] == 'Quarterly'

fullClassif = fullClassif[~mask]


'''

'''
mask = fullClassif['Commitment Duration'] == '6 M'

fullClassif = fullClassif[~mask]

maskDos = fullClassif['Commitment Duration'] == '1.25 YR'

fullClassif = fullClassif[~maskDos]

masTre = fullClassif['Commitment Duration'] == '1.42 YR'

fullClassif = fullClassif[~masTre]

masfour = fullClassif['Commitment Duration'] == '3.25 YR'

fullClassif = fullClassif[~masfour]

masfour = fullClassif['Commitment Duration'] == '7 YR'

fullClassif = fullClassif[~masfour]

masfour = fullClassif['Commitment Duration'] == '3.19 YR'

fullClassif = fullClassif[~masfour]

masfour = fullClassif['Commitment Duration'] == '1.22 YR'

fullClassif = fullClassif[~masfour]
'''
#test
training_labels1 = set(X_train['Commitment Duration'].unique())

training_labels2 = set(X_train['Commitment Duration2'].unique())
# Assuming 'fullClassif' is your DataFrame and 'unique_values_set' contains the values you want to filter out
mask = fullClassif['Commitment Duration'].isin(training_labels1)
fullClassif = fullClassif[mask]

mask2 = fullClassif['Commitment Duration2'].isin(training_labels2)
fullClassif = fullClassif[mask2]


training_labels1 = set(X_train['Billing frequency'].unique())

training_labels2 = set(X_train['Billing Frequency2'].unique())
# Assuming 'fullClassif' is your DataFrame and 'unique_values_set' contains the values you want to filter out
mask = fullClassif['Billing frequency'].isin(training_labels1)
fullClassif = fullClassif[mask]

mask2 = fullClassif['Billing Frequency2'].isin(training_labels2)
fullClassif = fullClassif[mask2]



#train_set, testo = train_test_split(fullClassif, test_size=0.2, random_state=42)
#get only trained on labels 
X_test = fullClassif.copy()  # Features in 'fullCat'
y_test = fullClassif.copy()


# Preprocessing categorical columns (if any)
label_encoders = {}
for col in X_train.select_dtypes(include='object').columns:
    label_encoders[col] = LabelEncoder()
    X_train[col] = label_encoders[col].fit_transform(X_train[col])

    
# Training the RandomForestClassifier model
rf_classifier = RandomForestClassifier(n_estimators=100)  # You can adjust parameters if needed
rf_classifier.fit(X_train, y_train)

'''
Model test
X_test = halfClassif.copy()
y_test = halfClassif.copy()
'''



# Preprocessing categorical columns in the test data (if any)
for col in X_test.select_dtypes(include='object').columns:
    X_test[col] = label_encoders[col].transform(X_test[col])  # Using the same encoders from training


# Predicting multiple columns in 'halfNotClassif'
predicted_classes = rf_classifier.predict(X_test)

print(predicted_classes[1:10])
#acc = accuracy_score(y_test, predicted_classes)
# Filling the predicted classes into 'halfNotClassif'

halfNotClassif[['Commitment Duration', 'Commitment Duration2', 'Billing frequency', 'Billing Frequency2', 'Consumption Model', 'Product type']] = predicted_classes[30582:]
#print('accuracy = ' + str(acc))
# 'halfNotClassif' DataFrame now contains the predicted classes in the respective columns

halfNotClassif.to_csv('testo.csv', quoting=csv.QUOTE_ALL)




#metrics 




