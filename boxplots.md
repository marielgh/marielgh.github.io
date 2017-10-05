
## School budgets  (multi-class-multi-label classification)

Objective is to predict the probability that a certain label is attached to a budget line item. Each row in the budget has mostly free-form text features, except for two that are noted as float. Any of the fields may or may not be empty.

**Features**

- FTE (float) - If an employee, the percentage of full-time that the employee works.
- Facility_or_Department - If expenditure is tied to a department/facility, that department/facility.
- Function_Description - A description of the function the expenditure was serving.
- Fund_Description - A description of the source of the funds.
- Job_Title_Description - If this is an employee, a description of that employee's job title.
- Location_Description - A description of where the funds were spent.
- Object_Description - A description of what the funds were used for.
- Position_Extra - Any extra information about the position that we have.
- Program_Description - A description of the program that the funds were used for.
- SubFund_Description - More detail on Fund_Description
- Sub_Object_Description - More detail on Object_Description
- Text_1 - Any additional text supplied by the district.
- Text_2 - Any additional text supplied by the district.
- Text_3 - Any additional text supplied by the district.
- Text_4 - Any additional text supplied by the district.
- Total (float) - The total cost of the expenditure.

**Labels**
    
Function:

- Aides Compensation
- Career & Academic Counseling
- Communications
- Curriculum Development
- Data Processing & Information Services
- Development & Fundraising
- Enrichment
- Extended Time & Tutoring
- Facilities & Maintenance
- Facilities Planning
- Finance, Budget, Purchasing & Distribution
- Food Services
- Governance
- Human Resources
- Instructional Materials & Supplies
- Insurance
- Legal
- Library & Media
- NO_LABEL
- Other Compensation
- Other Non-Compensation
- Parent & Community Relations
- Physical Health & Services
- Professional Development
- Recruitment
- Research & Accountability
- School Administration
- School Supervision
- Security & Safety
- Social & Emotional
- Special Population Program Management & Support
- Student Assignment
- Student Transportation
- Substitute Compensation
- Teacher Compensation
- Untracked Budget Set-Aside
- Utilities

Object_Type:

- Base Salary/Compensation
- Benefits
- Contracted Services
- Equipment & Equipment Lease
- NO_LABEL
- Other Compensation/Stipend
- Other Non-Compensation
- Rent/Utilities
- Substitute Compensation
- Supplies/Materials
- Travel & Conferences

Operating_Status:

- Non-Operating (If Non-Operating, all other labels must be NO LABEL)
- Operating, Not PreK-12
- PreK-12 Operating

Position_Type:

- (Exec) Director
- Area Officers
- Club Advisor/Coach
- Coordinator/Manager
- Custodian
- Guidance Counselor
- Instructional Coach
- Librarian
- NO_LABEL
- Non-Position
- Nurse
- Nurse Aide
- Occupational Therapist
- Other
- Physical Therapist
- Principal
- Psychologist
- School Monitor/Security
- Sec/Clerk/Other Admin
- Social Worker
- Speech Therapist
- Substitute
- TA
- Teacher
- Vice Principal

Pre_K:

- NO_LABEL
- Non PreK
- PreK

Reporting:

- NO_LABEL
- Non-School
- School

Sharing:

- Leadership & Management
- NO_LABEL
- School Reported
- School on Central Budgets
- Shared Services

Student_Type:

- Alternative
- At Risk
- ELL
- Gifted
- NO_LABEL
- Poverty
- PreK
- Special Education
- Unspecified

Use:

- Business Services
- ISPD
- Instruction
- Leadership
- NO_LABEL
- O&M
- Pupil Services & Enrichment
- Untracked Budget Set-Aside


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
```


```python
df_train = pd.read_csv('data/TrainingData.csv',index_col=0)
df_test = pd.read_csv('data/TestData.csv',index_col=0)
print(df_train.info())
print(df_test.info())
```

    /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2698: DtypeWarning: Columns (5,11) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)


    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 400277 entries, 134338 to 415831
    Data columns (total 25 columns):
    Function                  400277 non-null object
    Use                       400277 non-null object
    Sharing                   400277 non-null object
    Reporting                 400277 non-null object
    Student_Type              400277 non-null object
    Position_Type             400277 non-null object
    Object_Type               400277 non-null object
    Pre_K                     400277 non-null object
    Operating_Status          400277 non-null object
    Object_Description        375493 non-null object
    Text_2                    88217 non-null object
    SubFund_Description       306855 non-null object
    Job_Title_Description     292743 non-null object
    Text_3                    179964 non-null object
    Text_4                    53746 non-null object
    Sub_Object_Description    91603 non-null object
    Location_Description      162054 non-null object
    FTE                       126071 non-null float64
    Function_Description      342195 non-null object
    Facility_or_Department    53886 non-null object
    Position_Extra            264764 non-null object
    Total                     395722 non-null float64
    Program_Description       304660 non-null object
    Fund_Description          202877 non-null object
    Text_1                    292285 non-null object
    dtypes: float64(2), object(23)
    memory usage: 79.4+ MB
    None
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 50064 entries, 180042 to 249087
    Data columns (total 16 columns):
    Object_Description        48330 non-null object
    Program_Description       44811 non-null object
    SubFund_Description       16111 non-null object
    Job_Title_Description     32317 non-null object
    Facility_or_Department    2839 non-null object
    Sub_Object_Description    33612 non-null object
    Location_Description      37316 non-null object
    FTE                       19605 non-null float64
    Function_Description      46866 non-null object
    Position_Extra            13813 non-null object
    Text_4                    2814 non-null object
    Total                     49404 non-null float64
    Text_2                    4641 non-null object
    Text_3                    9486 non-null object
    Fund_Description          39586 non-null object
    Text_1                    15378 non-null object
    dtypes: float64(2), object(14)
    memory usage: 6.5+ MB
    None



```python
nolabels = ['FTE',
 'Total',
 'Object_Description',
 'SubFund_Description',
 'Job_Title_Description',
 'Sub_Object_Description',
 'Location_Description',
 'Function_Description',
 'Facility_or_Department',
 'Position_Extra',
 'Program_Description',
 'Fund_Description',
 'Text_1',
 'Text_2',
 'Text_3',
 'Text_4',
]
```


```python
labels = ['Function','Object_Type','Operating_Status','Position_Type','Pre_K','Reporting','Sharing','Student_Type','Use']
#nolabels = [c for c in df_train.columns if c not in labels]
X_train = df_train[nolabels].copy()
y_train = df_train[labels].copy()
X_test = df_test[nolabels].copy()
```


```python
((X_train.isnull().sum())/X_train.shape[0]*100).plot(kind='barh')
plt.xlabel('Percentage of missing values')
plt.show()
```


![png](images/boxplots/missing1.png)



```python
((X_test.isnull().sum())/X_test.shape[0]*100).plot(kind='barh')
plt.xlabel('Percentage of missing values')
plt.show()
```

![png](images/boxplots/missing2.png)


**y_train: dummy encoding of labels**


```python
y_train = pd.get_dummies(y_train,prefix_sep='__')
print(y_train.columns)
```

    Index(['Function__Aides Compensation',
           'Function__Career & Academic Counseling', 'Function__Communications',
           'Function__Curriculum Development',
           'Function__Data Processing & Information Services',
           'Function__Development & Fundraising', 'Function__Enrichment',
           'Function__Extended Time & Tutoring',
           'Function__Facilities & Maintenance', 'Function__Facilities Planning',
           ...
           'Student_Type__Special Education', 'Student_Type__Unspecified',
           'Use__Business Services', 'Use__ISPD', 'Use__Instruction',
           'Use__Leadership', 'Use__NO_LABEL', 'Use__O&M',
           'Use__Pupil Services & Enrichment', 'Use__Untracked Budget Set-Aside'],
          dtype='object', length=104)



```python
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=42)
```


```python
# Check that all labels are represented in the train set
y_tr.sum().min()
```




    24




```python
X_tr = X_tr.copy()
X_val = X_val.copy()
```

**Feature: FTE**


```python
X_tr.loc[:,'FTE'] = np.clip(X_tr['FTE'],0,1)
X_val.loc[:,'FTE'] = np.clip(X_val['FTE'],0,1)
X_test.loc[:,'FTE'] = np.clip(X_test['FTE'],0,1)
```

**Impute numeric columns and join text columns**


```python
from sklearn.preprocessing import Imputer
num_columns = ['FTE','Total']
text_columns = [c for c in X_tr.columns if c not in num_columns]

imputer = Imputer()
X_tr[num_columns]=imputer.fit_transform(X_tr[num_columns])

X_tr[text_columns] = X_tr[text_columns].fillna("")  
text_tr = X_tr[text_columns].apply(lambda x:" ".join(x), axis=1)
X_tr['text_data'] = text_tr
X_tr = X_tr.drop(text_columns, axis=1)
```


```python
X_val[num_columns]=imputer.transform(X_val[num_columns])

X_val[text_columns] = X_val[text_columns].fillna("")
text_val = X_val[text_columns].apply(lambda x:" ".join(x), axis=1)
X_val['text_data'] = text_val
X_val = X_val.drop(text_columns, axis=1)


X_test[num_columns]=imputer.transform(X_test[num_columns])

X_test[text_columns] = X_test[text_columns].fillna("")
text_test = X_test[text_columns].apply(lambda x:" ".join(x), axis=1)
X_test['text_data'] = text_test
X_test = X_test.drop(text_columns, axis=1)
```


```python
X_tr.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FTE</th>
      <th>Total</th>
      <th>text_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>444691</th>
      <td>0.420536</td>
      <td>218.450000</td>
      <td>EMPLOYEE BENEFITS  GENERAL FUND Teacher, Eleme...</td>
    </tr>
    <tr>
      <th>250816</th>
      <td>0.420536</td>
      <td>3126.332668</td>
      <td>Salaries And Wages For Teachers And Other Prof...</td>
    </tr>
    <tr>
      <th>393405</th>
      <td>0.420536</td>
      <td>674.710000</td>
      <td>EXTRA DUTY PAY-INSTRUCTIONAL     TEACHER, VE  ...</td>
    </tr>
    <tr>
      <th>299533</th>
      <td>0.420536</td>
      <td>286.688429</td>
      <td>Salaries And Wages For Teachers And Other Prof...</td>
    </tr>
    <tr>
      <th>357445</th>
      <td>0.420536</td>
      <td>221.843840</td>
      <td>ADDITIONAL/EXTRA DUTY PAY/STIP  FEDERAL GDPG F...</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_val.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FTE</th>
      <th>Total</th>
      <th>text_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>139668</th>
      <td>0.420536</td>
      <td>420.28</td>
      <td>EMPLOYER PD MED CONTRIBUTION  GENERAL FUND Tea...</td>
    </tr>
    <tr>
      <th>446779</th>
      <td>0.420536</td>
      <td>-1265.07</td>
      <td>CONTRA BENEFITS  GENERAL FUND Teacher Secondar...</td>
    </tr>
    <tr>
      <th>224958</th>
      <td>0.420536</td>
      <td>5320.20</td>
      <td>PURCHASED PROFESSIONAL AND TECH SVCS-OTHER FEE...</td>
    </tr>
    <tr>
      <th>386138</th>
      <td>0.420536</td>
      <td>1299.79</td>
      <td>SUPPLIES  SCHOOL-WIDE SCHOOL PGMS FOR TITLE GR...</td>
    </tr>
    <tr>
      <th>440032</th>
      <td>0.420536</td>
      <td>1145.50</td>
      <td>EMPLOYEE BENEFITS TRANSPORTATION GENERAL FUND ...</td>
    </tr>
  </tbody>
</table>
</div>



**NLP**


```python
import nltk
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
stem = nltk.stem.SnowballStemmer('english')

def nlp(s):
    s = s.replace( "K-", "K" ) # K-12 and K-8 
    tokenizer = RegexpTokenizer('(?u)\\b\\w\\w+\\b')
    tokens = tokenizer.tokenize(s.lower())
    #filtered_words = [word for word in tokens if word not in stopwords.words('english')]
    stem_words = [lem.lemmatize(word) for word in tokens]
    return " ".join(stem_words)
```


```python
t0 = dt.datetime.now()

X_tr['text_data'] = X_tr['text_data'].apply(nlp)
print('done NLP train')
X_val['text_data'] = X_val['text_data'].apply(nlp)
print('done NLP validation')
X_test['text_data'] = X_test['text_data'].apply(nlp)

t1 = dt.datetime.now()
print('\n'+str((t1 - t0).seconds/60)+' minutes')
```

    done NLP train
    done NLP validation
    
    0.9 minutes



```python
X_tr.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FTE</th>
      <th>Total</th>
      <th>text_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>444691</th>
      <td>0.420536</td>
      <td>218.450000</td>
      <td>employee benefit general fund teacher elementa...</td>
    </tr>
    <tr>
      <th>250816</th>
      <td>0.420536</td>
      <td>3126.332668</td>
      <td>salary and wage for teacher and other professi...</td>
    </tr>
    <tr>
      <th>393405</th>
      <td>0.420536</td>
      <td>674.710000</td>
      <td>extra duty pay instructional teacher ve school...</td>
    </tr>
    <tr>
      <th>299533</th>
      <td>0.420536</td>
      <td>286.688429</td>
      <td>salary and wage for teacher and other professi...</td>
    </tr>
    <tr>
      <th>357445</th>
      <td>0.420536</td>
      <td>221.843840</td>
      <td>additional extra duty pay stip federal gdpg fu...</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_val.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FTE</th>
      <th>Total</th>
      <th>text_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>139668</th>
      <td>0.420536</td>
      <td>420.28</td>
      <td>employer pd med contribution general fund teac...</td>
    </tr>
    <tr>
      <th>446779</th>
      <td>0.420536</td>
      <td>-1265.07</td>
      <td>contra benefit general fund teacher secondary ...</td>
    </tr>
    <tr>
      <th>224958</th>
      <td>0.420536</td>
      <td>5320.20</td>
      <td>purchased professional and tech svcs other fee...</td>
    </tr>
    <tr>
      <th>386138</th>
      <td>0.420536</td>
      <td>1299.79</td>
      <td>supply school wide school pgms for title grant...</td>
    </tr>
    <tr>
      <th>440032</th>
      <td>0.420536</td>
      <td>1145.50</td>
      <td>employee benefit transportation general fund b...</td>
    </tr>
  </tbody>
</table>
</div>



**Process text_data**


```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest

count_1gram = CountVectorizer(ngram_range=(1,1),token_pattern='(?u)\\b\\w\\w+\\b',stop_words='english',min_df=10)
Xt_tr = count_1gram.fit_transform(X_tr['text_data'])
features_1gram = np.array(count_1gram.get_feature_names())
print(features_1gram.shape)

with open("train_vocab_1gram.txt", "w") as output:
    for i in features_1gram:
        output.write(i+'\n')

count_train = CountVectorizer(ngram_range=(1,2),token_pattern='(?u)\\b\\w\\w+\\b',stop_words='english',min_df=10)
Xt_tr = count_train.fit_transform(X_tr['text_data'])
features_train = np.array(count_train.get_feature_names())
print(features_train.shape)

with open("train_vocab_2gram.txt", "w") as output:
    for i in features_train:
        output.write(i+'\n')

kbest = SelectKBest(chi2, k=1000)
Xt_tr = kbest.fit_transform(Xt_tr,y_tr.values)

features_train_selected = features_train[kbest.get_support()]
print(features_train_selected.shape)
print(features_train_selected)

with open("train_vocab_select.txt", "w") as output:
    for i in features_train_selected:
        output.write(i+'\n')
```

    (1954,)
    (12726,)
    (1000,)
    ['12' 'academic student' 'accounting' 'acq' 'acquisition' 'activiti'
     'activiti cocurricular' 'activity' 'activity non' 'activity related'
     'activity transportation' 'additional' 'additional extra' 'addl'
     'addl regular' 'admin' 'admin school' 'admin service' 'administration'
     'administration year' 'administrative' 'administrative support'
     'administrator' 'administrator legal' 'administrator support' 'adult'
     'adult voc' 'advisor' 'advisor high' 'afterschool' 'afterschool program'
     'aide' 'allotment' 'allotment teacher' 'allowance'
     'alternative opportunity' 'alternative school' 'arra' 'arra education'
     'arra professional' 'arra stimulus' 'art sport' 'assessment'
     'assessment instructional' 'assessment research' 'assessment specialist'
     'assignment' 'assignment regular' 'assignment team' 'assistance'
     'assistance esea' 'assistance title' 'assistant' 'assistant clinic'
     'assistant nurse' 'assistant principal' 'asst' 'asst high'
     'asst principal' 'athletic' 'athletic supplement' 'athletics'
     'athletics pupil' 'athletics sport' 'athletics student' 'attorney'
     'attorney non' 'bachelor' 'bachelor title' 'based' 'based ece' 'basic'
     'basic educational' 'basic program' 'benefit' 'benefit custodial'
     'benefit employee' 'benefit food' 'benefit general' 'benefit human'
     'benefit itemgd' 'benefit personnel' 'benefit safety' 'benefit teacher'
     'benefit transportation' 'bilingual' 'bilingual education' 'blank'
     'blank regular' 'bldg' 'bldg service' 'board' 'board education' 'bond'
     'bonding' 'bonding cost' 'book' 'book periodical' 'book textbook'
     'breakfast' 'breakfast lunch' 'budget' 'budget school' 'building' 'bus'
     'bus compute' 'bus driver' 'cafeteria' 'cafeteria employee' 'campus'
     'campus overtime' 'campus payroll' 'campus security' 'capital'
     'capital reserve' 'care' 'care upkeep' 'center' 'center educational'
     'center general' 'central' 'certificated' 'certificated employee'
     'certificated salary' 'certified' 'certified regular'
     'certified substitute' 'chair' 'chair supp' 'charter' 'charter basic'
     'charter school' 'child' 'child nutrition' 'child targeted' 'childhood'
     'childhood education' 'childhood regular' 'choice' 'choice non'
     'classroom supply' 'clerical' 'clinic' 'clinic aide' 'clinic assistant'
     'coach' 'coach athletics' 'coach instructional' 'coaching'
     'coaching supplement' 'cocurricular' 'cocurricular ed' 'communication'
     'community' 'community relation' 'community service' 'comp' 'comp general'
     'comp title' 'compute' 'computer' 'computer technology' 'construction'
     'contra' 'contra benefit' 'contracted' 'conversion' 'conversion charter'
     'cooperating' 'cooperating teacher' 'coordinator' 'cost insurance'
     'counsel' 'counsel school' 'counseling' 'counselor' 'counselor high'
     'craft' 'craft trade' 'critical' 'critical need' 'cu' 'cu super'
     'curricular' 'curricular spo' 'curriculum' 'curriculum development'
     'curriculum teacher' 'custodial' 'custodial department' 'custodial helper'
     'custodial school' 'custodian' 'custodian asst' 'custodian operation'
     'custodian regular' 'customer' 'customer relation' 'day' 'day school'
     'degreed' 'degreed substitute' 'department' 'department bus'
     'department chair' 'department cu' 'department police' 'deputy general'
     'dev' 'dev eval' 'dev human' 'development' 'development service'
     'director' 'director undistributed' 'disability' 'disadvantaged'
     'disadvantaged child' 'disadvantaged youth' 'district'
     'district objective' 'district special' 'district wide' 'driver'
     'driver reg' 'driver transportation' 'driver undistributed' 'driver yr'
     'duty' 'duty pay' 'early' 'early childhood' 'early education' 'ece'
     'ece paraprofessional' 'ece professional' 'ed' 'ed activity'
     'ed allotment' 'ed art' 'ed opportunity' 'ed para' 'education'
     'education adult' 'education itinerant' 'education ot' 'education pt'
     'education regular' 'education special' 'education stabilitazion'
     'educational' 'educational medium' 'educational service' 'effectiveness'
     'ela' 'ela general' 'ela teaching' 'electricity' 'elementary'
     'elementary education' 'elementary regular' 'elementary spec' 'elpa'
     'elpa eng' 'elpa preschool' 'employee' 'employee adult'
     'employee allowance' 'employee assessment' 'employee benefit'
     'employee custodial' 'employee food' 'employee general'
     'employee miscellaneous' 'employee personnel' 'employee safety'
     'employee salary' 'employee teacher' 'employee transportation'
     'end allocation' 'eng' 'eng lang' 'equipment' 'equipment bus' 'eval'
     'eval general' 'extended' 'extended day' 'extended week' 'extended year'
     'extra' 'extra curricular' 'extra duty' 'extracurricular'
     'extracurricular activity' 'extracurricular supp' 'facilitator'
     'facilitator elem' 'facilitator sec' 'facility' 'facility acq'
     'facility maintenance' 'facsimile' 'facsimile service' 'federal'
     'federal gdpg' 'fee' 'field' 'field trip' 'food' 'food nutrition'
     'food preparation' 'food serv' 'food service' 'fund' 'fund blank'
     'fund bus' 'fund campus' 'fund child' 'fund cooperating' 'fund custodial'
     'fund custodian' 'fund early' 'fund ela' 'fund elementary'
     'fund facilitator' 'fund field' 'fund food' 'fund fy' 'fund growth'
     'fund library' 'fund manager' 'fund nurse' 'fund occupational'
     'fund overtime' 'fund physical' 'fund principal' 'fund psychologist'
     'fund regular' 'fund risk' 'fund secretary' 'fund security' 'fund social'
     'fund speech' 'fund student' 'fund supt' 'fund teacher' 'fy'
     'fy facilitator' 'fy occupational' 'fy speech' 'garage' 'garage non'
     'gdpg' 'gdpg fund' 'general' 'general administration' 'general assignment'
     'general counsel' 'general elementary' 'general fund' 'general ledger'
     'general operating' 'general purpose' 'general supply' 'gifted'
     'gifted talented' 'grade' 'grade program' 'grant service' 'growth'
     'growth planning' 'guard' 'guidance' 'guidance service' 'health'
     'health service' 'helper' 'helper day' 'high' 'human' 'human resource'
     'humanity' 'idea' 'ii' 'ii regular' 'improvement' 'improvement instr'
     'improving' 'improving basic' 'information' 'information service' 'inst'
     'inst staff' 'instr' 'instr serv' 'instruction' 'instruction campus'
     'instruction certificated' 'instruction curriculum'
     'instruction instruction' 'instruction primary' 'instruction regular'
     'instruction sped' 'instruction substitute' 'instructional'
     'instructional early' 'instructional general' 'instructional gifted'
     'instructional guidance' 'instructional instructional'
     'instructional itemgb' 'instructional literacy' 'instructional male'
     'instructional nursing' 'instructional psychological'
     'instructional social' 'instructional special' 'instructional staff'
     'insurance' 'insurance bonding' 'itemgb' 'itemgd' 'itemgd preschool'
     'itemgg' 'itinerant' 'itinerant professional' 'k12 conversion'
     'k12 general' 'lang prof' 'language' 'language general'
     'language pathologst' 'leadership' 'leadership campus' 'learning'
     'learning leadership' 'ledger' 'ledger accounting' 'legal'
     'legal department' 'legal service' 'librarian' 'librarian elem' 'library'
     'library medium' 'library service' 'library tech' 'literacy'
     'literacy coach' 'lunch' 'lunch program' 'maint' 'maint plant'
     'maintenance' 'maintenance operating' 'maintenance operation' 'male'
     'male athletics' 'man' 'man undistributed' 'manager' 'manager food'
     'material' 'material community' 'math' 'math science' 'medium'
     'medium center' 'medium service' 'midd' 'midd sch' 'middle' 'mild'
     'mild moderate' 'misc' 'misc general' 'misc school' 'miscellaneous'
     'miscellaneous admin' 'moderate' 'monitoring' 'monitoring service'
     'multilingual' 'multilingual outreach' 'national' 'national school'
     'natural gas' 'need' 'new' 'new teacher' 'non' 'non certificated'
     'non project' 'non public' 'nurse' 'nurse general' 'nurse regular'
     'nurse school' 'nurse service' 'nursing' 'nursing service' 'nutrition'
     'nutrition cafeteria' 'nutrition fund' 'nutrition misc' 'nutrition reg'
     'nutrition service' 'objective' 'objective salary' 'occupational'
     'occupational therapist' 'office' 'office administrative'
     'office principal' 'office superintendent' 'officer' 'officer extra'
     'officer regular' 'officer resource' 'officer safety' 'officer sub'
     'operating' 'operating fund' 'operating misc' 'operation'
     'operation child' 'operation clerical' 'operation computer'
     'operation counselor' 'operation custodial' 'operation custodian'
     'operation maint' 'operation plant' 'operation security'
     'operation student' 'operation teacher' 'operation transportation'
     'operation utility' 'opp' 'opportunity' 'opportunity arra'
     'opportunity school' 'ot' 'ot special' 'outreach' 'outreach general'
     'outreach state' 'overtime' 'overtime support' 'para' 'para regular'
     'paraprofessional' 'paraprofessional early' 'paraprofessional general'
     'paraprofessional special' 'parent' 'participant' 'participant teacher'
     'pathologst' 'pathologst special' 'patrol' 'patrol man' 'pay'
     'pay overtime' 'pay public' 'pay smoothed' 'pay stip' 'payroll'
     'payroll degreed' 'payroll principal' 'payroll teacher' 'periodical'
     'periodical medium' 'personal' 'personal service' 'personnel'
     'personnel child' 'personnel operation' 'personnel school'
     'personnel service' 'personnel unallocated' 'personnel undistributed'
     'physical' 'physical therapist' 'planning' 'planning dev' 'plant'
     'plant general' 'police' 'police patrol' 'pool' 'pool basic' 'pool salary'
     'pre' 'preparation' 'preparation serving' 'preschool' 'preschool prog'
     'preschool program' 'primary' 'primary grade' 'principal' 'principal asst'
     'principal elementary' 'principal undistributed' 'prof act' 'professi'
     'professi operation' 'professi school' 'professi state' 'professi title'
     'professional' 'professional information' 'professional instructional'
     'professional operation' 'professional security' 'professional state'
     'professional tech' 'professional technical' 'prog'
     'prog paraprofessional' 'prog professional' 'program' 'program 12'
     'program early' 'program general' 'program instruction'
     'program miscellaneous' 'program summer' 'program teacher' 'project'
     'project administrator' 'project craft' 'project office'
     'project professional' 'project undesignated' 'psychological'
     'psychological service' 'psychologist' 'psychologist psychological'
     'psychologist regular' 'psychologist special' 'pt special' 'public'
     'public school' 'public utility' 'pupil' 'pupil activity'
     'pupil transportation' 'purchase' 'purchase equipment' 'purchased'
     'purchased professional' 'purchased service' 'purpose' 'purpose school'
     'reading writing' 'record' 'recruitment' 'recruitment district'
     'recruitment professional' 'reg' 'reg transportation' 'registration'
     'regular' 'regular ela' 'regular elpa' 'regular employee'
     'regular extended' 'regular general' 'regular instruction' 'regular non'
     'regular pay' 'regular special' 'regular state' 'regular title'
     'regular tuition' 'regular unalloc' 'related' 'related state' 'relation'
     'relation department' 'repair' 'repair maintenance' 'reqd misc' 'research'
     'research ela' 'research general' 'reserve' 'reserve fund' 'resource'
     'resource new' 'resource non' 'resource officer' 'resource recruitment'
     'retirement' 'retrd' 'retrd shrt' 'revenue' 'revenue fund' 'run'
     'run extra' 'safety' 'safety security' 'salary' 'salary regular'
     'salary time' 'salary wage' 'sch' 'school' 'school admin'
     'school administration' 'school arra' 'school attorney' 'school breakfast'
     'school choice' 'school counselor' 'school facility' 'school food'
     'school general' 'school guidance' 'school instruction'
     'school leadership' 'school library' 'school nurse' 'school nutrition'
     'school principal' 'school professional' 'school resource'
     'school security' 'school service' 'schoolwide' 'schoolwide activity'
     'science' 'sea' 'second' 'second run' 'secondary' 'secondary high'
     'secretary' 'secretary ii' 'secretary regular' 'security'
     'security department' 'security general' 'security monitoring'
     'security non' 'security officer' 'security operation' 'security safety'
     'security security' 'security service' 'serv' 'serv undistributed'
     'service' 'service business' 'service care' 'service central'
     'service charter' 'service child' 'service community' 'service district'
     'service employee' 'service food' 'service fund' 'service improvement'
     'service instructional' 'service medium' 'service miscellaneous'
     'service non' 'service operation' 'service police' 'service pupil'
     'service school' 'service security' 'service special' 'service student'
     'service sub' 'service substitute' 'service support' 'service teacher'
     'service title' 'service transportation' 'service worker' 'serving'
     'serving food' 'severe' 'sewer' 'short' 'short term' 'shrt' 'shrt term'
     'skill' 'skill specialist' 'smoothed' 'social' 'social work'
     'social worker' 'spec' 'spec ed' 'special' 'special ed'
     'special education' 'special instruction' 'special revenue'
     'special trust' 'specialist' 'sped' 'speech' 'speech language'
     'speech therapist' 'spo' 'sport' 'sport activiti' 'sport sea'
     'stabilitazion' 'staff' 'staff dev' 'staff service' 'staff training'
     'state' 'state comp' 'state ed' 'state elpa' 'state gifted' 'stimulus'
     'stimulus teacher' 'stip' 'stip assessment' 'stip athletics' 'stip human'
     'stip teacher' 'stipend' 'student' 'student activity' 'student advisor'
     'student assessment' 'student service' 'student transportation' 'sub'
     'sub general' 'sub human' 'sub regular' 'sub teacher'
     'subsistence employee' 'substitute' 'substitute basic' 'substitute pool'
     'substitute professional' 'substitute teacher' 'summer' 'summer ed'
     'sunday' 'sunday pay' 'super' 'super sub' 'super undistributed'
     'superintendent' 'superintendent general' 'supp' 'supp extra' 'supplement'
     'supply' 'supply community' 'supply material' 'supply medium'
     'supply primary' 'supply school' 'support' 'support personnel'
     'support school' 'support service' 'support support' 'support team' 'supt'
     'supt instructional' 'svcs' 'svcs fee' 'talented' 'talented general'
     'talented professional' 'talented regular' 'talented state' 'targeted'
     'targeted assistance' 'teacher' 'teacher bachelor' 'teacher certified'
     'teacher effectiveness' 'teacher elementary' 'teacher instruction'
     'teacher learning' 'teacher librarian' 'teacher professi'
     'teacher recruitment' 'teacher regular' 'teacher retirement'
     'teacher retrd' 'teacher school' 'teacher secondary' 'teacher short'
     'teacher stipend' 'teacher sub' 'teacher substitute' 'teacher super'
     'teaching' 'team' 'team general' 'team non' 'tech' 'tech svcs' 'technical'
     'technical service' 'technology' 'technology specialist' 'telephone'
     'telephone facsimile' 'temp' 'term' 'term sub' 'textbook'
     'textbook periodical' 'therapist' 'therapist special' 'tif' 'time'
     'time employee' 'title' 'title arra' 'title disadvantaged' 'title ii'
     'title improving' 'title schoolwide' 'trade' 'trade service' 'training'
     'training instructional' 'training svcs' 'transportation'
     'transportation bus' 'transportation department' 'transportation general'
     'transportation operation' 'transportation pupil' 'transportation second'
     'transportation service' 'transportation transportation' 'travel'
     'travel allowance' 'travel employee' 'travel registration' 'trip' 'trust'
     'tuition' 'tuition based' 'unalloc' 'unalloc budget' 'unallocated'
     'unallocated extracurricular' 'unallocated security'
     'unallocated transportation' 'undesignated' 'undesignated care'
     'undesignated employee' 'undesignated general' 'undistributed'
     'undistributed general' 'undistributed national'
     'undistributed transportation' 'upkeep' 'upkeep bldg' 'utility'
     'utility general' 'utility local' 'voc' 'voc ed' 'wage'
     'wage disadvantaged' 'wage substitute' 'wage support' 'wage teacher'
     'water' 'week' 'week school' 'wide resource' 'work' 'work service'
     'worker' 'worker ii' 'worker regular' 'worker social'
     'worker undistributed' 'workshop' 'workshop participant' 'year' 'youth'
     'youth title' 'yr' 'yr garage']


**find similar**


```python
from sklearn.feature_extraction.text import CountVectorizer

count_val = CountVectorizer(ngram_range=(1,1),token_pattern='(?u)\\b\\w\\w+\\b',stop_words='english',min_df=10)
Xt_val = count_val.fit_transform(X_val['text_data'])
features_val = np.array(count_val.get_feature_names())
print(features_val.shape)

features_new_val = set(features_val)-set(features_1gram)
print(len(features_new_val))
print(features_new_val)

with open("val_vs_train.txt", "w") as output:
    for i in features_new_val:
        output.write(i+'\n')

#selected_features_missing = set(features_train_selected)-set(features_val)
#print(len(selected_features_missing))
#print(selected_features_missing)
```

    (1361,)
    1
    {'provide'}



```python
import difflib

def find_similar_val(s):
    tokenizer = RegexpTokenizer('(?u)\\b\\w\\w+\\b')
    tokens = tokenizer.tokenize(s)
    tokens2 = []
    for word in tokens:
        if word in features_new_val:
            word = " ".join(difflib.get_close_matches(word,features_1gram,1,0.8))
        tokens2.append(word)
    return " ".join(tokens2)
```


```python
sim_list = []
with open("similar_val.txt", "w") as output:
    not_found_val = 0
    for i in features_new_val:
        sim_word = " ".join(difflib.get_close_matches(i,features_1gram,1,0.8))
        sim_list.append(sim_word)
        output.write(i+" "+sim_word+'\n')
        if not sim_word:
            not_found_val = not_found_val + 1 

print(sim_list)
print(list(set(sim_list) & set(features_train_selected)))
```

    ['provided']
    []



```python
X_val['text_data'] = X_val['text_data'].apply(find_similar_val)
```

**


```python
from sklearn.feature_extraction.text import CountVectorizer

count_test = CountVectorizer(ngram_range=(1,1),token_pattern='(?u)\\b\\w\\w+\\b',stop_words='english',min_df=10)
Xt_test = count_test.fit_transform(X_test['text_data'])
features_test = np.array(count_test.get_feature_names())
print(features_test.shape)

features_new_test = set(features_test)-set(features_train)
print(len(features_new_test))
print(features_new_test)

with open("test_vs_train.txt", "w") as output:
    for i in features_new_test:
        output.write(i+'\n')
                
#selected_features_missing = set(features_train_selected)-set(features_test)
#print(len(selected_features_missing))
#print(selected_features_missing)
```

    (1145,)
    140
    {'nwsppr', 'sc', 'cl', 'chem', 'beh', 'inclus', 'alarm', 'abc', 'occup', 'house', '6th', 'im', 'faci', 'theater', 'footbl', 'digital', 'proc', 'consulting', 'theat', 'cell', 'abd', 'overcrowding', 'diagnosticia', 'meal', 'tv', 'chi', 'spe', 'cate', 'vfb', 'mariachi', 'grds', 'nondisciplinary', 'volleyb', 'ther', 'astii', 'lan', 'dose', 'cdr', 'wastewater', 'spiii', 'bb', 'se', 'services', 'au', 'specia', 'score', 'lssp', 'struggler', 'geography', 'dramatics', 'volly', 'pri', 'inn', 'detention', 'ad', 'supprt', 'ahc', 'adapt', 'vh', 'bil', 'publication', 'cor', 'theatre', 'baccalaureate', 'trainr', 'equity', 'sb', 'cycle', 'phone', 'jr', 'tm', 'netwrk', 'srvc', 'ssig', 'aep', 'volbl', 'yrbk', 'trustee', 'unacceptable', 'ely', 'prod', 'laundry', 'honor', 'busn', 'independence', 'pal', 'cch', 'wrest', 'trainee', 'agricul', 'acctg', 'fac', 'op', 'includes', 'footbll', 'uil', 'architect', 'jrvar', 'government', 'delta', 'bsktbl', 'commercial', 'attend', 'analy', 'mn', 'signing', 'fr', 'kin', 'sk', 'nat', 'emotional', 'carry', 'housekeeping', 'groundskeeper', 'money', 'ni', 'reproduction', 'badge', 'ml', 'nvfb', 'journalism', 'dram', 'priority', 'courier', 'used', 'ah', 'coun', 'room', 'hc', 'dropout', 'refreshment', 'expend', 'ofcr', 'culinary', 'varfb', 'englsh', 'facil', 'demonstration', 'fto', 'baseb'}



```python
import difflib

def find_similar_test(s):
    tokenizer = RegexpTokenizer('(?u)\\b\\w\\w+\\b')
    tokens = tokenizer.tokenize(s)
    tokens2 = []
    for word in tokens:
        if word in features_new_test:
            word = " ".join(difflib.get_close_matches(word,features_1gram,1,0.8))
        tokens2.append(word)
    return " ".join(tokens2)
```


```python
X_test['text_data'] = X_test['text_data'].apply(find_similar_test)
```


```python
Xt_val = count_train.transform(X_val['text_data'])
Xt_test = count_train.transform(X_test['text_data'])

Xt_val = kbest.transform(Xt_val)
Xt_test = kbest.transform(Xt_test)

print(Xt_tr.shape)
print(Xt_val.shape)
print(Xt_test.shape)
```

    (320221, 1000)
    (80056, 1000)
    (50064, 1000)


**Combine numeric and text data**


```python
from scipy import sparse
from scipy.sparse import hstack

X_tr_num = sparse.csr_matrix(X_tr[num_columns].values)
X_tr = hstack([Xt_tr,X_tr_num])

X_val_num = sparse.csr_matrix(X_val[num_columns].values)
X_val = hstack([Xt_val,X_val_num])

X_test_num = sparse.csr_matrix(X_test[num_columns].values)
X_test = hstack([Xt_test,X_test_num])
```

    (320221, 1002)
    (80056, 1002)
    (50064, 1002)


**MaxAbsScaler**


```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```


```python
print(X_tr.shape)
print(y_tr.shape)

print(X_val.shape)
print(y_val.shape)

print(X_test.shape)
```

    (320221, 1002)
    (320221, 104)
    (80056, 1002)
    (80056, 104)
    (50064, 1002)


**Modelling**


```python
import xgboost as xgb
from sklearn.externals import joblib

t0 = dt.datetime.now()
    
b=[0,37,48,51,76,79,82,87,96,104]
scores=[]
pred_prob_val = np.ones((X_val.shape[0],104))
pred_prob_test = np.ones((X_test.shape[0],104))

for i in np.arange(9):

    print('\n\nRound '+str(i+1)+':\n\n')
    
    w = b[i+1]-b[i]
    opt_params = {'booster' : 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric': 'mlogloss',
              'learning_rate': 0.2,
              'n_estimators':1000,
              'colsample_bytree': 0.3,
              'max_depth':5,
              'min_child_weight':32,
              'reg_lambda':1,
              'subsample':0.9,
              'n_jobs':1,
              'silent':True,
              'num_class' : w}
    
    y_tr_i = y_tr.iloc[:,b[i]:b[i+1]].values
    y_val_i = y_val.iloc[:,b[i]:b[i+1]].values
    y_tr_i = np.argmax(y_tr_i, axis=1)
    y_val_i = np.argmax(y_val_i, axis=1)
                     
    dtrain = xgb.DMatrix(X_tr, label=y_tr_i)
    dvalid = xgb.DMatrix(X_val, label=y_val_i)
    dtest = xgb.DMatrix(X_test)
                     
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
                     
    model_xgb = xgb.train(opt_params, dtrain, 200, watchlist, early_stopping_rounds=20, 
                          maximize=False, verbose_eval=10)
                     
    pred_prob_test[:,b[i]:b[i+1]] = model_xgb.predict(dtest,ntree_limit=model_xgb.best_ntree_limit).reshape(X_test.shape[0],w)
    pred_prob_val[:,b[i]:b[i+1]] = model_xgb.predict(dvalid,ntree_limit=model_xgb.best_ntree_limit).reshape(X_val.shape[0],w)
    scores.append(model_xgb.best_score)
    joblib.dump(model_xgb, 'xgb_less_sim_2gram'+str(i+1)+'.model')
    
t1 = dt.datetime.now()

print(opt_params)

print(scores)

print('\n'+str((t1 - t0).seconds/60)+' minutes')
```

    
    
    Round 1:
    
    
    [0]	train-mlogloss:1.83189	valid-mlogloss:1.8447
    Multiple eval metrics have been passed: 'valid-mlogloss' will be used for early stopping.
    
    Will train until valid-mlogloss hasn't improved in 20 rounds.
    [10]	train-mlogloss:0.695993	valid-mlogloss:0.703255
    [20]	train-mlogloss:0.462728	valid-mlogloss:0.468333
    [30]	train-mlogloss:0.373847	valid-mlogloss:0.37963
    [40]	train-mlogloss:0.323967	valid-mlogloss:0.329918
    [50]	train-mlogloss:0.294816	valid-mlogloss:0.300973
    [60]	train-mlogloss:0.272547	valid-mlogloss:0.279186
    [70]	train-mlogloss:0.257236	valid-mlogloss:0.264195
    [80]	train-mlogloss:0.245271	valid-mlogloss:0.2528
    [90]	train-mlogloss:0.234834	valid-mlogloss:0.242691
    [100]	train-mlogloss:0.225999	valid-mlogloss:0.234173
    [110]	train-mlogloss:0.218523	valid-mlogloss:0.227144
    [120]	train-mlogloss:0.212077	valid-mlogloss:0.221059
    [130]	train-mlogloss:0.206698	valid-mlogloss:0.21604
    [140]	train-mlogloss:0.201852	valid-mlogloss:0.211608
    [150]	train-mlogloss:0.197371	valid-mlogloss:0.207527
    [160]	train-mlogloss:0.193158	valid-mlogloss:0.20364
    [170]	train-mlogloss:0.189474	valid-mlogloss:0.200267
    [180]	train-mlogloss:0.186184	valid-mlogloss:0.197283
    [190]	train-mlogloss:0.183197	valid-mlogloss:0.194778
    [199]	train-mlogloss:0.180511	valid-mlogloss:0.192463
    
    
    Round 2:
    
    
    [0]	train-mlogloss:1.71356	valid-mlogloss:1.71445
    Multiple eval metrics have been passed: 'valid-mlogloss' will be used for early stopping.
    
    Will train until valid-mlogloss hasn't improved in 20 rounds.
    [10]	train-mlogloss:0.51052	valid-mlogloss:0.51381
    [20]	train-mlogloss:0.283994	valid-mlogloss:0.287166
    [30]	train-mlogloss:0.203949	valid-mlogloss:0.206559
    [40]	train-mlogloss:0.16047	valid-mlogloss:0.162981
    [50]	train-mlogloss:0.134928	valid-mlogloss:0.137256
    [60]	train-mlogloss:0.117562	valid-mlogloss:0.119893
    [70]	train-mlogloss:0.106343	valid-mlogloss:0.108755
    [80]	train-mlogloss:0.098102	valid-mlogloss:0.100658
    [90]	train-mlogloss:0.091924	valid-mlogloss:0.094463
    [100]	train-mlogloss:0.08695	valid-mlogloss:0.089491
    [110]	train-mlogloss:0.082835	valid-mlogloss:0.085541
    [120]	train-mlogloss:0.07973	valid-mlogloss:0.08257
    [130]	train-mlogloss:0.076884	valid-mlogloss:0.079867
    [140]	train-mlogloss:0.074152	valid-mlogloss:0.077248
    [150]	train-mlogloss:0.071887	valid-mlogloss:0.075129
    [160]	train-mlogloss:0.069972	valid-mlogloss:0.073379
    [170]	train-mlogloss:0.067973	valid-mlogloss:0.071514
    [180]	train-mlogloss:0.066306	valid-mlogloss:0.069886
    [190]	train-mlogloss:0.064885	valid-mlogloss:0.068582
    [199]	train-mlogloss:0.063557	valid-mlogloss:0.067384
    
    
    Round 3:
    
    
    [0]	train-mlogloss:0.869076	valid-mlogloss:0.869651
    Multiple eval metrics have been passed: 'valid-mlogloss' will be used for early stopping.
    
    Will train until valid-mlogloss hasn't improved in 20 rounds.
    [10]	train-mlogloss:0.210947	valid-mlogloss:0.21284
    [20]	train-mlogloss:0.116852	valid-mlogloss:0.118408
    [30]	train-mlogloss:0.090686	valid-mlogloss:0.091979
    [40]	train-mlogloss:0.078855	valid-mlogloss:0.080028
    [50]	train-mlogloss:0.072176	valid-mlogloss:0.073395
    [60]	train-mlogloss:0.06763	valid-mlogloss:0.068735
    [70]	train-mlogloss:0.064361	valid-mlogloss:0.065412
    [80]	train-mlogloss:0.061189	valid-mlogloss:0.062174
    [90]	train-mlogloss:0.05898	valid-mlogloss:0.059972
    [100]	train-mlogloss:0.057177	valid-mlogloss:0.058201
    [110]	train-mlogloss:0.055187	valid-mlogloss:0.056235
    [120]	train-mlogloss:0.053674	valid-mlogloss:0.054717
    [130]	train-mlogloss:0.052336	valid-mlogloss:0.053367
    [140]	train-mlogloss:0.05121	valid-mlogloss:0.052274
    [150]	train-mlogloss:0.050078	valid-mlogloss:0.051187
    [160]	train-mlogloss:0.049006	valid-mlogloss:0.050152
    [170]	train-mlogloss:0.048225	valid-mlogloss:0.049409
    [180]	train-mlogloss:0.047369	valid-mlogloss:0.048521
    [190]	train-mlogloss:0.046669	valid-mlogloss:0.047868
    [199]	train-mlogloss:0.046074	valid-mlogloss:0.047313
    
    
    Round 4:
    
    
    [0]	train-mlogloss:1.81125	valid-mlogloss:1.81307
    Multiple eval metrics have been passed: 'valid-mlogloss' will be used for early stopping.
    
    Will train until valid-mlogloss hasn't improved in 20 rounds.
    [10]	train-mlogloss:0.552062	valid-mlogloss:0.554879
    [20]	train-mlogloss:0.307052	valid-mlogloss:0.309735
    [30]	train-mlogloss:0.225357	valid-mlogloss:0.227455
    [40]	train-mlogloss:0.184634	valid-mlogloss:0.186465
    [50]	train-mlogloss:0.162026	valid-mlogloss:0.163958
    [60]	train-mlogloss:0.147728	valid-mlogloss:0.149749
    [70]	train-mlogloss:0.135598	valid-mlogloss:0.137862
    [80]	train-mlogloss:0.127325	valid-mlogloss:0.129841
    [90]	train-mlogloss:0.120695	valid-mlogloss:0.123422
    [100]	train-mlogloss:0.114898	valid-mlogloss:0.117693
    [110]	train-mlogloss:0.110466	valid-mlogloss:0.113422
    [120]	train-mlogloss:0.106526	valid-mlogloss:0.109667
    [130]	train-mlogloss:0.103268	valid-mlogloss:0.106579
    [140]	train-mlogloss:0.100305	valid-mlogloss:0.103794
    [150]	train-mlogloss:0.097507	valid-mlogloss:0.101175
    [160]	train-mlogloss:0.095114	valid-mlogloss:0.098945
    [170]	train-mlogloss:0.093045	valid-mlogloss:0.097068
    [180]	train-mlogloss:0.091057	valid-mlogloss:0.095257
    [190]	train-mlogloss:0.089385	valid-mlogloss:0.093739
    [199]	train-mlogloss:0.087787	valid-mlogloss:0.092311
    
    
    Round 5:
    
    
    [0]	train-mlogloss:0.867108	valid-mlogloss:0.86706
    Multiple eval metrics have been passed: 'valid-mlogloss' will be used for early stopping.
    
    Will train until valid-mlogloss hasn't improved in 20 rounds.
    [10]	train-mlogloss:0.192972	valid-mlogloss:0.193516
    [20]	train-mlogloss:0.096586	valid-mlogloss:0.096969
    [30]	train-mlogloss:0.071443	valid-mlogloss:0.071777
    [40]	train-mlogloss:0.059419	valid-mlogloss:0.059565
    [50]	train-mlogloss:0.051957	valid-mlogloss:0.051986
    [60]	train-mlogloss:0.04686	valid-mlogloss:0.046802
    [70]	train-mlogloss:0.043325	valid-mlogloss:0.043278
    [80]	train-mlogloss:0.040881	valid-mlogloss:0.0408
    [90]	train-mlogloss:0.038726	valid-mlogloss:0.038674
    [100]	train-mlogloss:0.036999	valid-mlogloss:0.036998
    [110]	train-mlogloss:0.035732	valid-mlogloss:0.035792
    [120]	train-mlogloss:0.034726	valid-mlogloss:0.034868
    [130]	train-mlogloss:0.033799	valid-mlogloss:0.03402
    [140]	train-mlogloss:0.033036	valid-mlogloss:0.033347
    [150]	train-mlogloss:0.03234	valid-mlogloss:0.032698
    [160]	train-mlogloss:0.031824	valid-mlogloss:0.032259
    [170]	train-mlogloss:0.031277	valid-mlogloss:0.031777
    [180]	train-mlogloss:0.030778	valid-mlogloss:0.031326
    [190]	train-mlogloss:0.030419	valid-mlogloss:0.031032
    [199]	train-mlogloss:0.030025	valid-mlogloss:0.030696
    
    
    Round 6:
    
    
    [0]	train-mlogloss:0.954778	valid-mlogloss:0.955099
    Multiple eval metrics have been passed: 'valid-mlogloss' will be used for early stopping.
    
    Will train until valid-mlogloss hasn't improved in 20 rounds.
    [10]	train-mlogloss:0.437242	valid-mlogloss:0.438067
    [20]	train-mlogloss:0.313599	valid-mlogloss:0.314242
    [30]	train-mlogloss:0.254341	valid-mlogloss:0.254987
    [40]	train-mlogloss:0.220908	valid-mlogloss:0.221693
    [50]	train-mlogloss:0.199904	valid-mlogloss:0.200768
    [60]	train-mlogloss:0.184353	valid-mlogloss:0.185367
    [70]	train-mlogloss:0.173047	valid-mlogloss:0.174017
    [80]	train-mlogloss:0.163112	valid-mlogloss:0.164025
    [90]	train-mlogloss:0.154151	valid-mlogloss:0.155072
    [100]	train-mlogloss:0.147522	valid-mlogloss:0.14858
    [110]	train-mlogloss:0.141823	valid-mlogloss:0.142757
    [120]	train-mlogloss:0.136913	valid-mlogloss:0.137833
    [130]	train-mlogloss:0.131553	valid-mlogloss:0.132553
    [140]	train-mlogloss:0.127283	valid-mlogloss:0.128321
    [150]	train-mlogloss:0.123489	valid-mlogloss:0.124493
    [160]	train-mlogloss:0.120327	valid-mlogloss:0.121382
    [170]	train-mlogloss:0.117566	valid-mlogloss:0.11866
    [180]	train-mlogloss:0.114854	valid-mlogloss:0.115947
    [190]	train-mlogloss:0.11164	valid-mlogloss:0.112748
    [199]	train-mlogloss:0.109665	valid-mlogloss:0.110794
    
    
    Round 7:
    
    
    [0]	train-mlogloss:1.33327	valid-mlogloss:1.3335
    Multiple eval metrics have been passed: 'valid-mlogloss' will be used for early stopping.
    
    Will train until valid-mlogloss hasn't improved in 20 rounds.
    [10]	train-mlogloss:0.569265	valid-mlogloss:0.568829
    [20]	train-mlogloss:0.401161	valid-mlogloss:0.400819
    [30]	train-mlogloss:0.331677	valid-mlogloss:0.331406
    [40]	train-mlogloss:0.289329	valid-mlogloss:0.288929
    [50]	train-mlogloss:0.263074	valid-mlogloss:0.262719
    [60]	train-mlogloss:0.244071	valid-mlogloss:0.243621
    [70]	train-mlogloss:0.229372	valid-mlogloss:0.228851
    [80]	train-mlogloss:0.217543	valid-mlogloss:0.217054
    [90]	train-mlogloss:0.207569	valid-mlogloss:0.20711
    [100]	train-mlogloss:0.198864	valid-mlogloss:0.198477
    [110]	train-mlogloss:0.192391	valid-mlogloss:0.192096
    [120]	train-mlogloss:0.186143	valid-mlogloss:0.185908
    [130]	train-mlogloss:0.180806	valid-mlogloss:0.180689
    [140]	train-mlogloss:0.175709	valid-mlogloss:0.175692
    [150]	train-mlogloss:0.171196	valid-mlogloss:0.171183
    [160]	train-mlogloss:0.166892	valid-mlogloss:0.166954
    [170]	train-mlogloss:0.162734	valid-mlogloss:0.162839
    [180]	train-mlogloss:0.15927	valid-mlogloss:0.159593
    [190]	train-mlogloss:0.155926	valid-mlogloss:0.156388
    [199]	train-mlogloss:0.152871	valid-mlogloss:0.153392
    
    
    Round 8:
    
    
    [0]	train-mlogloss:1.65386	valid-mlogloss:1.65543
    Multiple eval metrics have been passed: 'valid-mlogloss' will be used for early stopping.
    
    Will train until valid-mlogloss hasn't improved in 20 rounds.
    [10]	train-mlogloss:0.529184	valid-mlogloss:0.53424
    [20]	train-mlogloss:0.319581	valid-mlogloss:0.325238
    [30]	train-mlogloss:0.246304	valid-mlogloss:0.252363
    [40]	train-mlogloss:0.210337	valid-mlogloss:0.216095
    [50]	train-mlogloss:0.18524	valid-mlogloss:0.190788
    [60]	train-mlogloss:0.169223	valid-mlogloss:0.174726
    [70]	train-mlogloss:0.155227	valid-mlogloss:0.160643
    [80]	train-mlogloss:0.145104	valid-mlogloss:0.150405
    [90]	train-mlogloss:0.136828	valid-mlogloss:0.142209
    [100]	train-mlogloss:0.129524	valid-mlogloss:0.134875
    [110]	train-mlogloss:0.123576	valid-mlogloss:0.128932
    [120]	train-mlogloss:0.118346	valid-mlogloss:0.123676
    [130]	train-mlogloss:0.113878	valid-mlogloss:0.11912
    [140]	train-mlogloss:0.110354	valid-mlogloss:0.115557
    [150]	train-mlogloss:0.1065	valid-mlogloss:0.111701
    [160]	train-mlogloss:0.103781	valid-mlogloss:0.109081
    [170]	train-mlogloss:0.100798	valid-mlogloss:0.106205
    [180]	train-mlogloss:0.098499	valid-mlogloss:0.103994
    [190]	train-mlogloss:0.096311	valid-mlogloss:0.101824
    [199]	train-mlogloss:0.09444	valid-mlogloss:0.099932
    
    
    Round 9:
    
    
    [0]	train-mlogloss:1.63653	valid-mlogloss:1.63994
    Multiple eval metrics have been passed: 'valid-mlogloss' will be used for early stopping.
    
    Will train until valid-mlogloss hasn't improved in 20 rounds.
    [10]	train-mlogloss:0.643074	valid-mlogloss:0.651387
    [20]	train-mlogloss:0.433549	valid-mlogloss:0.442204
    [30]	train-mlogloss:0.348896	valid-mlogloss:0.357313
    [40]	train-mlogloss:0.301434	valid-mlogloss:0.309171
    [50]	train-mlogloss:0.27382	valid-mlogloss:0.281547
    [60]	train-mlogloss:0.254183	valid-mlogloss:0.262071
    [70]	train-mlogloss:0.239157	valid-mlogloss:0.247
    [80]	train-mlogloss:0.227133	valid-mlogloss:0.234914
    [90]	train-mlogloss:0.217452	valid-mlogloss:0.225287
    [100]	train-mlogloss:0.209509	valid-mlogloss:0.217383
    [110]	train-mlogloss:0.201956	valid-mlogloss:0.209807
    [120]	train-mlogloss:0.195389	valid-mlogloss:0.203268
    [130]	train-mlogloss:0.188818	valid-mlogloss:0.196734
    [140]	train-mlogloss:0.183668	valid-mlogloss:0.191683
    [150]	train-mlogloss:0.179504	valid-mlogloss:0.187592
    [160]	train-mlogloss:0.174763	valid-mlogloss:0.182827
    [170]	train-mlogloss:0.170457	valid-mlogloss:0.178577
    [180]	train-mlogloss:0.166788	valid-mlogloss:0.174985
    [190]	train-mlogloss:0.163264	valid-mlogloss:0.171625
    [199]	train-mlogloss:0.160436	valid-mlogloss:0.168922
    {'booster': 'gbtree', 'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'learning_rate': 0.2, 'n_estimators': 1000, 'colsample_bytree': 0.3, 'max_depth': 5, 'min_child_weight': 32, 'reg_lambda': 1, 'subsample': 0.9, 'n_jobs': 1, 'silent': True, 'num_class': 8}
    [0.192463, 0.067384, 0.047313, 0.092311, 0.030696, 0.110794, 0.153392, 0.099932, 0.168922]
    
    123.71666666666667 minutes


**multi multi log loss**


```python
from multiloss import *

print(multimultiloss(y_val.values,pred_prob_val))
print(multimultiloss(y_val.values,pred_prob_val2))
```

    [ 0.19246346  0.06738434  0.04731264  0.09231058  0.03069647  0.11079395
      0.15339215  0.09993167  0.16892221]
    0.107023050371
    [ 0.1924602   0.06738393  0.04731239  0.0923099   0.03069594  0.1107928
      0.15339097  0.09993061  0.16891682]
    0.107021508256

