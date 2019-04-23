import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from array import array
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.preprocessing import LabelEncoder
import numpy as geek 
import math
gapminder_csv_url ='http://bit.ly/2cLzoxH'
import pickle
import requests
import json
from sklearn.externals import joblib


# Read data from file 'filename.csv'
data=pd.read_csv('C:/Users/Cookie Monster/Desktop/Dataset.csv') 
data=data.drop(['Unnamed: 0','Highest Score', 'City'],axis='columns')
data.columns = data.columns.str.replace('University Name','University')
data.columns = data.columns.str.replace('Lowest Score','ScoreBoundry')
data["Major"]= data["Major"].str.replace("\r\r\r\r\n", "", case = False) 
data["Major"]= data["Major"].str.replace("Management Informationsystems", "Management Information Systems", case = False)
data["Major"]= data["Major"].str.replace("' ", "'", case = False)
data["Major"]= data["Major"].str.replace("  ", " ", case = False)
data["Major"]= data["Major"].str.replace(" '", "'", case = False)

fields = ['SAY', 'SOZ', 'DIL', 'EA']
majors=['Architecture', 'Computer Engineering',
       'Electrical and Electronics Engineering', 'Industrial Engineering',
       'Civil Engineering', 'Mechanical Engineering',
       'Management Information Systems', 'Interpretership',
       ' School Science Teaching', 'Arabic Teaching', 'Art History',
       'Biology', 'Guidance and Psychological Counseling', 'History',
       'Math', 'Midwifery', 'Nursing', 'Pre-school Teaching',
       'Primary School Math Teaching', 'Social Science Teaching',
       'Teaching', 'Turkish Language and Literature', 'Turkish Teaching',
       'Biomedical Engineering', 'Electrical Engineering',
       'Islamic Sciences', 'Mechatronic Engineering',
       'Molecular Biology and Genetics', 'Nutrition and Dietetics',
       'Physical Therapy and Rehabilitation', 'Cinema and Television',
       'Contemporary Turkish Dialects and Literatures', 'Finance',
       'Gastronomy and Kitchen Arts', 'Health Management', 'Law',
       'Map Engineering', 'Medicine',
       'Metallurgy and Materials Engineering', 'Sociology',
       'Tourism Guidance', 'Veterinary', 'International Relations',
       'Agricultural Economy', 'Archaeology', 'Chemistry', 'Dentistry',
       'Econometric', 'English Language and Literature',
       'English Teaching', 'Food Engineering', 'Geography',
       'German Language and Literature', 'International Relations ',
       'Journalism', 'Philosophy', 'Psychology',
       'Public Relations and Promotion', 'Radio Television and Cinema',
       'Science Teaching', 'Space Sciences and Technologies', 'Theology',
       'Tourism Business', 'City and Region Planning',
       'Communication Design and Management', 'French Teaching',
       'German Teaching', 'Language and Speech Therapy',
       'Public Relations and Advertising',
       'Russian Language and Literature', 'Social Service',
       'Agricultural Machinery and Technologies Engineering',
       'American Culture and Literature', 'Anthropology',
       'Arabic Language and Literature', 'Astronomy and Space Science ',
       'Clasic Archaeology', 'Computer Teaching',
       'French Language and Literature', 'Hittitology', 'Hungarology',
       'Italian Language and Literature',
       'Japanese Language and Literature',
       'Korean Language and Literature', 'Landscape Architecture',
       'Latin Language and Literature', 'Linguistic', 'Livestock',
       'Modern Greek Language and Literature',
       'Modern Turkish Dialect and Literature',
       'Persian Language and Literature', 'Pharmacy', 'Physics',
       'Political Science and Public Administration', 'Sinology',
       'Spanish Language and Literature', 'Statistics',
       'Water Product Engineering', 'Geography Teaching', 'Math Teaching',
       'Public Administration', 'Biology Teaching', 'Chemistry Teaching',
       'Computer Science', 'Environmental Engineering', 'History ',
       'History Teaching', 'International Business and Trade',
       'Interpretation-Interpretership', 'Logistic Management',
       'Movie Design and Management',
       'Physical Therapy and Rehabilitation ', 'Physics Teaching',
       'Advertising',
       'Agricultural Machinery and Technologies Engineering ',
       'Bioengineering', 'Chemistry Engineering', 'Political Science',
       'Industrial Products Design', 'Nutrition and Dietetics ',
       'Philosophy Teaching', 'Automotive Engineering',
       'Medicine Engineering', 'Spor Management',
       'Energy System Engineering', 'Interior Architecture',
       'Software Engineering', 'Chemisty']

print("(1)SAY (2)SOZ (3)DIL (4)EA")
print("Choose a test type")
test_type = int(input())

if test_type== 1:
  test="SAY"
elif test_type== 2:
  test="SOZ"
elif test_type== 3:
  test="DIL"
elif test_type== 4:
  test="EA"
else:
  print("Please choose one") 


print('''
(0)'Architecture',
(1)'Computer Engineering',
(2)'Electrical and Electronics Engineering'
(3)'Industrial Engineering',
(4)'Civil Engineering',
(5)'Mechanical Engineering',
(6)'Management Information Systems',
(7)'Interpretership',
(8)' School Science Teaching',
(9)'Arabic Teaching',
(10)'Art History',
(11)'Biology', 
(12)'Guidance and Psychological Counseling',
(13)'History',
(14)'Math',
(15)'Midwifery', 
(16)'Nursing', 
(17)'Pre-school Teaching',
(18)'Primary School Math Teaching',
(19)'Social Science Teaching',
(20)'Teaching', 
(21)'Turkish Language and Literature',
(22)'Turkish Teaching',
(23)'Biomedical Engineering', 
(24)'Electrical Engineering',
(25)'Islamic Sciences',
(26)'Mechatronic Engineering',
(27)'Molecular Biology and Genetics',
(28)'Nutrition and Dietetics',
(29)'Physical Therapy and Rehabilitation', 
(30)'Cinema and Television',
(31)'Contemporary Turkish Dialects and Literatures',
(32)'Finance',
(33)'Gastronomy and Kitchen Arts', 
(34)'Health Management', 
(35)'Law',
(36)'Map Engineering', 
(37)'Medicine',
(38)'Metallurgy and Materials Engineering', 
(39)'Sociology',
(40)'Tourism Guidance',
(41)'Veterinary', 
(42)'International Relations',
(43)'Agricultural Economy',
(44)'Archaeology', 
(45)'Chemistry', 
(46)'Dentistry',
(47)'Econometric', 
(48)'English Language and Literature',
(49)'English Teaching', 
(50)'Food Engineering',
(51)'Geography',
(52)'German Language and Literature',
(53)'International Relations ',
(54)'Journalism', 
(55)'Philosophy', 
(56)'Psychology',
(57)'Public Relations and Promotion',
(58)'Radio Television and Cinema',
(59)'Science Teaching', 
(60)'Space Sciences and Technologies', 
(61)'Theology',
(62)'Tourism Business',
(63)'City and Region Planning',
(64)'Communication Design and Management',
(65)'French Teaching',
(66)'German Teaching', 
(67)'Language and Speech Therapy',
(68)'Public Relations and Advertising',
(69)'Russian Language and Literature', 
(70)'Social Service',
(71)'Agricultural Machinery and Technologies Engineering',
(72)'American Culture and Literature',
(73)'Anthropology',
(74)'Arabic Language and Literature',
(75)'Astronomy and Space Science ',
(76)'Clasic Archaeology', 
(77)'Computer Teaching',
(78)'French Language and Literature',
(79)'Hittitology',
(80)'Hungarology',
(81)'Italian Language and Literature',
(82)'Japanese Language and Literature',
(83)'Korean Language and Literature',
(84)'Landscape Architecture',
(85)'Latin Language and Literature',
(86)'Linguistic',
(87)'Livestock',
(88)'Modern Greek Language and Literature',
(89)'Modern Turkish Dialect and Literature',
(90)'Persian Language and Literature', 
(91)'Pharmacy',
(92)'Physics',
(93)'Political Science and Public Administration', 
(94)'Sinology',
(95)'Spanish Language and Literature',
(96)'Statistics',
(97)'Water Product Engineering', 
(98)'Geography Teaching', 
(99)'Math Teaching',
(100)'Public Administration', 
(101)'Biology Teaching',
(102)'Chemistry Teaching',
(103)'Computer Science', 
(104)'Environmental Engineering',
(105)'History ',
(106)'History Teaching', 
(107)'International Business and Trade',
(108)'Interpretation-Interpretership',
(109)'Logistic Management',
(110)'Movie Design and Management',
(111)'Physical Therapy and Rehabilitation ',
(112)'Physics Teaching',
(113)'Advertising',
(114)'Agricultural Machinery and Technologies Engineering ',
(115)'Bioengineering', 
(116)'Chemistry Engineering',
(117)'Political Science',
(118)'Industrial Products Design',
(119)'Nutrition and Dietetics ',
(120)'Philosophy Teaching',
(121)'Automotive Engineering',
(122)'Medicine Engineering', 
(123)'Spor Management',
(124)'Energy System Engineering',
(125)'Interior Architecture',
(126)'Software Engineering', 
(127)'Chemisty' ''')

print()
print()
print("Choose a major")
major_no = int(input())

print("plz enter your exam score")
e = int(input())
print("plz enter your ranking")
r= int(input())
a=str(majors[major_no])

dataset=data[(data['Field'].eq(test)==True) & (data['Major'].eq(a)==True)]
dataset=dataset.sort_values(['University','YEAR'],ascending=True)
#show the data that are needed for new table
dataset_training=dataset.drop(['Field','Region','Major'],axis='columns')
#how many results we have
rowCount= dataset_training.shape[0]
result = pd.DataFrame({ "University":[], 
                        "Score":[],
                        "Ranking":[],
                        "Year":[]})
#loop for data model & new table
k=0
row=rowCount
l=0
X1=pd.DataFrame()
X11=pd.DataFrame()
X22=[]
count=0
while (k<row):
    stop=l+4
    while (l<stop):       
    #    X1 =pd.DataFrame(dataset_training.iloc[:k])    
         l=l+1
         k=k+1
    X1 =pd.DataFrame(dataset_training.iloc[k-4:k])
    X2=X1.values
    arr=np.array([[0,0,0,2019]])
    X2=np.vstack([X2, arr])    

    
    # Use only one feature
    data_X = X2[:, np.newaxis, 3]
    data_Y = X2[:, np.newaxis, 2]
    # Split the data into training/testing sets
    data_X_train = data_X[:-1]
    data_X_test = data_X[-1:]

    # Split the targets into training/testing sets
    data_y_train = data_Y[:-1]
    data_y_test = data_Y[-1:]

    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(data_X_train, data_y_train)

     # Make predictions using the testing set
    data_y_pred=regr.predict(data_X_test)[0]
    #print(data_y_pred)
    
    t=int(data_y_pred[0])
    X2[4][0]=X2[0][0]
    X2[4][2]=t
    X1=X11
    #store in result
    add_values= pd.DataFrame({"University":[X2[4][0]], 
                          "Score":[X2[4][1]],
                          "Ranking":[X2[4][2]],
                          "Year":[X2[4][3]]}) 
    result=result.append(add_values)
    #print(X2)
    #X2=np.delete(X2, np.s_[0:5:], axis=0)
    #X2=X3
    count=count+1

result = result.drop("Score", axis=1)   
final_result = pd.DataFrame({ "University":[],
                        "Ranking":[],
                        "Year":[],
                        "Distance":[]})
value=result.values   
inc=0
while (inc<count):       
    if r<value[inc][1]:
        distance = (r-value[inc][1])
        addn_values= pd.DataFrame({"University":[value[inc][0]], 
                          "Ranking":[value[inc][1]],
                          "Year":[value[inc][2]],
                          "Distance":[distance]}) 
        final_result=final_result.append(addn_values)
    else:pass
        
    inc=inc+1
final_result=final_result.sort_values(by=['Distance'])
final_result=final_result.drop(['Distance'],axis='columns')
           
print()
print()
joblib.dump(final_result, 'model.pkl')
model=joblib.load('model.pkl')
print(model)