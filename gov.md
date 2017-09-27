
https://public.enigma.com/browse/u-s-government-spending-contracts/1a932abc-8398-47ff-ad33-d1eb9a8958cc


```python
import pandas as pd
import numpy as np
```


```python
cols = ['maj_agency_cat','unique_transaction_id','base_and_all_options_value','base_and_exercised_options_value']
```


```python
df00 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2000.csv',usecols=cols)
df01 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2001.csv',usecols=cols)
df02 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2002.csv',usecols=cols)
df03 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2003.csv',usecols=cols)
df04 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2004.csv',usecols=cols)
df05 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2005.csv',usecols=cols)
df06 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2006.csv',usecols=cols)
df07 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2007.csv',usecols=cols)
df08 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2008.csv',usecols=cols)
```


```python
df09 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2009.csv',usecols=cols)
df10 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2010.csv',usecols=cols)
df11 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2011.csv',usecols=cols)
df12 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2012.csv',usecols=cols)
df13 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2013.csv',usecols=cols)
df14 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2014.csv',usecols=cols)
df15 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2015.csv',usecols=cols)
df16 = pd.read_csv('/Users/mariel/gov_data/GovernmentSpendingContracts-2016.csv',usecols=cols)
```


```python
list_df = [df00,df01,df02,df03,df04,df05,df06,df07,df08,df09,df10,df11,df12,df13,df14,df15,df16]
```


```python
for i,df in enumerate(list_df):
    s = df.base_and_exercised_options_value.sum()
    y = np.arange(2000,2017)
    print('{}, {}, {} trillion'.format(y[i],s,s/1e12))
```

    2000, 21935659668.309998, 0.021935659668309997 trillion
    2001, 841011122108.43, 0.8410111221084301 trillion
    2002, 356142442981.55, 0.35614244298155 trillion
    2003, 417583198225.62, 0.41758319822562 trillion
    2004, 129380036430.88002, 0.12938003643088003 trillion
    2005, 1142073439501.1006, 1.1420734395011005 trillion
    2006, 757891949998.02, 0.75789194999802 trillion
    2007, 1292849178086.5378, 1.2928491780865379 trillion
    2008, 213397213061.43, 0.21339721306143 trillion
    2009, 480232037303.0702, 0.4802320373030702 trillion
    2010, 72551213451.79002, 0.07255121345179003 trillion
    2011, 759985575708.4503, 0.7599855757084503 trillion
    2012, 570799514555.7997, 0.5707995145557997 trillion
    2013, 216167210710.21005, 0.21616721071021006 trillion
    2014, 25332277616.979996, 0.025332277616979995 trillion
    2015, 156640921303.42007, 0.15664092130342008 trillion
    2016, 501511070642.5205, 0.5015110706425205 trillion



```python
dict_agency = pd.read_csv('dict_agency2.csv')
dict_agency['name_agency'] = dict_agency['name_agency'].apply(lambda x: str(x).upper())
dict_agency = dict_agency.set_index('ind_agency').to_dict()['name_agency']
```


```python
dep2 = ['8000','8900','7500','3600','1400','1600','1500','7000','9700','1300','7200']
list_df = [df00,df01,df02,df03,df04,df05,df06,df07,df08,df09,df10,df11,df12,df13,df14,df15,df16]
list_g2 = []
for df in list_df:
    df = df[df.base_and_exercised_options_value>=0]
    g_cat = df.groupby('maj_agency_cat').base_and_exercised_options_value.sum().reset_index()
    g_cat['ind_agency'] = g_cat.maj_agency_cat.apply(lambda x: x.split(':')[0])
    g_cat['name_agency'] = g_cat.maj_agency_cat.apply(lambda x: x.split(':')[1])
    g_cat = g_cat.groupby('ind_agency').base_and_exercised_options_value.sum().sort_values(ascending=False).reset_index()
    g_cat = g_cat[g_cat.ind_agency != ''].copy()
    g_cat['percentage'] = g_cat.base_and_exercised_options_value/(g_cat.base_and_exercised_options_value.sum())*100
    g_cat = g_cat.drop('base_and_exercised_options_value',axis=1)
    g_cat = g_cat.set_index('ind_agency')
    g_cat = g_cat.loc[dep2]
    g_cat = g_cat.reset_index()
    g_cat['name_agency'] = g_cat['ind_agency'].apply(lambda x: dict_agency[x])
    g_cat = g_cat[['name_agency','percentage']].set_index('name_agency')
    list_g2.append(g_cat)
```


```python
list_g2[12]
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
      <th>percentage</th>
    </tr>
    <tr>
      <th>name_agency</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NATIONAL AERONAUTICS AND SPACE ADMINISTRATION</th>
      <td>1.444371</td>
    </tr>
    <tr>
      <th>DEPARTMENT OF ENERGY</th>
      <td>4.474475</td>
    </tr>
    <tr>
      <th>DEPARTMENT OF HEALTH AND HUMAN SERVICES</th>
      <td>4.175027</td>
    </tr>
    <tr>
      <th>DEPARTMENT OF VETERANS AFFAIRS</th>
      <td>2.110476</td>
    </tr>
    <tr>
      <th>DEPARTMENT OF THE INTERIOR</th>
      <td>0.574522</td>
    </tr>
    <tr>
      <th>DEPARTMENT OF LABOR</th>
      <td>0.937695</td>
    </tr>
    <tr>
      <th>DEPARTMENT OF JUSTICE</th>
      <td>0.820120</td>
    </tr>
    <tr>
      <th>DEPARTMENT OF HOMELAND SECURITY</th>
      <td>1.755065</td>
    </tr>
    <tr>
      <th>DEPARTMENT OF DEFENSE</th>
      <td>76.936166</td>
    </tr>
    <tr>
      <th>DEPARTMENT OF COMMERCE</th>
      <td>0.297971</td>
    </tr>
    <tr>
      <th>AGENCY FOR INTERNATIONAL DEVELOPMENT</th>
      <td>0.532830</td>
    </tr>
  </tbody>
</table>
</div>




```python
g_cat = pd.concat(list_g2,axis=1)
g_cat = g_cat.fillna(0)
g_cat = g_cat.transpose()
```


```python
g_cat.index=np.arange(2000,2017)
```


```python
g_cat
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
      <th>name_agency</th>
      <th>NATIONAL AERONAUTICS AND SPACE ADMINISTRATION</th>
      <th>DEPARTMENT OF ENERGY</th>
      <th>DEPARTMENT OF HEALTH AND HUMAN SERVICES</th>
      <th>DEPARTMENT OF VETERANS AFFAIRS</th>
      <th>DEPARTMENT OF THE INTERIOR</th>
      <th>DEPARTMENT OF LABOR</th>
      <th>DEPARTMENT OF JUSTICE</th>
      <th>DEPARTMENT OF HOMELAND SECURITY</th>
      <th>DEPARTMENT OF DEFENSE</th>
      <th>DEPARTMENT OF COMMERCE</th>
      <th>AGENCY FOR INTERNATIONAL DEVELOPMENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000</th>
      <td>28.773158</td>
      <td>28.282812</td>
      <td>20.435956</td>
      <td>0.125481</td>
      <td>0.148846</td>
      <td>0.073936</td>
      <td>1.689803</td>
      <td>0.007141</td>
      <td>11.934709</td>
      <td>0.831711</td>
      <td>1.622980</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>0.728477</td>
      <td>1.756945</td>
      <td>95.644375</td>
      <td>0.017535</td>
      <td>0.012905</td>
      <td>0.010625</td>
      <td>0.012698</td>
      <td>0.000856</td>
      <td>1.338312</td>
      <td>0.105915</td>
      <td>0.075387</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>1.464194</td>
      <td>1.188632</td>
      <td>88.800265</td>
      <td>0.042193</td>
      <td>0.050169</td>
      <td>0.091197</td>
      <td>0.019232</td>
      <td>0.128818</td>
      <td>3.480622</td>
      <td>0.244774</td>
      <td>3.511632</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>4.587340</td>
      <td>4.686131</td>
      <td>76.877866</td>
      <td>0.109150</td>
      <td>0.137722</td>
      <td>0.107640</td>
      <td>0.327304</td>
      <td>0.238242</td>
      <td>8.120653</td>
      <td>0.828849</td>
      <td>1.794585</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>13.072901</td>
      <td>8.521964</td>
      <td>6.122738</td>
      <td>11.696123</td>
      <td>3.161667</td>
      <td>0.985425</td>
      <td>1.754215</td>
      <td>13.235085</td>
      <td>10.542899</td>
      <td>3.841001</td>
      <td>3.838430</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>0.591847</td>
      <td>14.126021</td>
      <td>3.142314</td>
      <td>0.631529</td>
      <td>0.318179</td>
      <td>0.129067</td>
      <td>71.609611</td>
      <td>4.588226</td>
      <td>1.128384</td>
      <td>0.174733</td>
      <td>0.424714</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>2.061107</td>
      <td>5.074494</td>
      <td>13.542313</td>
      <td>1.537408</td>
      <td>0.767924</td>
      <td>0.291390</td>
      <td>0.827533</td>
      <td>9.717587</td>
      <td>52.381504</td>
      <td>1.765666</td>
      <td>2.213656</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>0.440244</td>
      <td>0.484348</td>
      <td>2.864578</td>
      <td>0.286476</td>
      <td>0.097247</td>
      <td>0.056773</td>
      <td>0.290329</td>
      <td>0.566865</td>
      <td>90.793517</td>
      <td>1.747137</td>
      <td>1.246055</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>1.462755</td>
      <td>3.168053</td>
      <td>5.736717</td>
      <td>1.592376</td>
      <td>0.506351</td>
      <td>0.229678</td>
      <td>0.827530</td>
      <td>2.382598</td>
      <td>74.281727</td>
      <td>0.317430</td>
      <td>4.221258</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>2.442842</td>
      <td>7.113560</td>
      <td>8.554694</td>
      <td>2.632171</td>
      <td>0.722306</td>
      <td>0.047130</td>
      <td>2.216939</td>
      <td>3.648505</td>
      <td>58.020219</td>
      <td>0.709768</td>
      <td>5.279927</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>0.093719</td>
      <td>0.013641</td>
      <td>0.038376</td>
      <td>9.648830</td>
      <td>4.614445</td>
      <td>16.411108</td>
      <td>30.973187</td>
      <td>0.058415</td>
      <td>3.003001</td>
      <td>5.687331</td>
      <td>0.321962</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>1.547276</td>
      <td>2.414546</td>
      <td>3.853196</td>
      <td>2.439372</td>
      <td>0.582140</td>
      <td>0.823993</td>
      <td>1.164309</td>
      <td>2.576310</td>
      <td>75.784794</td>
      <td>0.366519</td>
      <td>1.169686</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>1.444371</td>
      <td>4.474475</td>
      <td>4.175027</td>
      <td>2.110476</td>
      <td>0.574522</td>
      <td>0.937695</td>
      <td>0.820120</td>
      <td>1.755065</td>
      <td>76.936166</td>
      <td>0.297971</td>
      <td>0.532830</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>2.912411</td>
      <td>12.085830</td>
      <td>11.021698</td>
      <td>8.374809</td>
      <td>1.586972</td>
      <td>3.284687</td>
      <td>3.076021</td>
      <td>4.472089</td>
      <td>30.436530</td>
      <td>1.084086</td>
      <td>2.674509</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>0.097513</td>
      <td>0.039514</td>
      <td>0.353169</td>
      <td>12.248496</td>
      <td>13.496518</td>
      <td>10.445530</td>
      <td>15.464883</td>
      <td>0.699602</td>
      <td>12.565341</td>
      <td>9.195122</td>
      <td>0.050781</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>2.734155</td>
      <td>3.416638</td>
      <td>7.980521</td>
      <td>23.658282</td>
      <td>3.071111</td>
      <td>1.568338</td>
      <td>5.037616</td>
      <td>3.540709</td>
      <td>15.998625</td>
      <td>2.080984</td>
      <td>6.420377</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>3.128543</td>
      <td>5.274874</td>
      <td>3.640473</td>
      <td>4.463973</td>
      <td>0.957567</td>
      <td>0.508757</td>
      <td>1.443014</td>
      <td>2.728741</td>
      <td>66.122132</td>
      <td>0.817371</td>
      <td>1.058904</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
g_cat.plot.bar(stacked=True,figsize=(15,5),colormap='Paired')
ax = plt.subplot(111)
chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
ax.legend(loc='right', bbox_to_anchor=(1.8, 0.5), shadow=True, ncol=1,prop={'size': 12})
#plt.legend(loc = 'lower left', prop={'size': 13})
plt.ylabel('% of budget')
plt.show()
```


![png](output_13_0.png)



```python

```
