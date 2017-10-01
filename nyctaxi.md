
## New York City Taxi Trip Duration 

### Exploratory Data Analysis

<div style="background-color:white">
    
    ssss
    
</div>
    


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-darkgrid')
```

```python
train_df = pd.read_csv('train.csv',parse_dates=["pickup_datetime"])
test_df = pd.read_csv('test.csv',parse_dates=["pickup_datetime"])
print(train_df.shape)
train_df.head()
```

    (1458644, 10)





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
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>trip_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id2875421</td>
      <td>2</td>
      <td>2016-03-14 17:24:55</td>
      <td>1</td>
      <td>-73.982155</td>
      <td>40.767937</td>
      <td>-73.964630</td>
      <td>40.765602</td>
      <td>N</td>
      <td>455</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id2377394</td>
      <td>1</td>
      <td>2016-06-12 00:43:35</td>
      <td>1</td>
      <td>-73.980415</td>
      <td>40.738564</td>
      <td>-73.999481</td>
      <td>40.731152</td>
      <td>N</td>
      <td>663</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id3858529</td>
      <td>2</td>
      <td>2016-01-19 11:35:24</td>
      <td>1</td>
      <td>-73.979027</td>
      <td>40.763939</td>
      <td>-74.005333</td>
      <td>40.710087</td>
      <td>N</td>
      <td>2124</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id3504673</td>
      <td>2</td>
      <td>2016-04-06 19:32:31</td>
      <td>1</td>
      <td>-74.010040</td>
      <td>40.719971</td>
      <td>-74.012268</td>
      <td>40.706718</td>
      <td>N</td>
      <td>429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id2181028</td>
      <td>2</td>
      <td>2016-03-26 13:30:55</td>
      <td>1</td>
      <td>-73.973053</td>
      <td>40.793209</td>
      <td>-73.972923</td>
      <td>40.782520</td>
      <td>N</td>
      <td>435</td>
    </tr>
  </tbody>
</table>
</div>

<br>
```python
train_df.info()
```
<pre style="background-color:white"><code><class 'pandas.core.frame.DataFrame'>
RangeIndex: 1458644 entries, 0 to 1458643
Data columns (total 10 columns):
id                    1458644 non-null object
vendor_id             1458644 non-null int64
pickup_datetime       1458644 non-null datetime64[ns]
passenger_count       1458644 non-null int64
pickup_longitude      1458644 non-null float64
pickup_latitude       1458644 non-null float64
dropoff_longitude     1458644 non-null float64
dropoff_latitude      1458644 non-null float64
store_and_fwd_flag    1458644 non-null object
trip_duration         1458644 non-null int64
dtypes: datetime64[ns](1), float64(4), int64(3), object(2)
memory usage: 111.3+ MB</code></pre>

**Data fields**
- id - a unique identifier for each trip
- vendor_id - a code indicating the provider associated with the trip record
- pickup_datetime - date and time when the meter was engaged
- passenger_count - the number of passengers in the vehicle (driver entered value)
- pickup_longitude - the longitude where the meter was engaged
- pickup_latitude - the latitude where the meter was engaged
- dropoff_longitude - the longitude where the meter was disengaged
- dropoff_latitude - the latitude where the meter was disengaged
- store_and_fwd_flag - this flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
- trip_duration - duration of the trip in seconds

**Vendor id**


```python
train_df['vendor_id'].value_counts().sort_index().plot(kind='bar')
plt.xticks(rotation='horizontal')
plt.ylabel('Frequency')
plt.xlabel('Vendor')
plt.show()

train_df['vendor_id'].value_counts().sort_index()/sum(train_df['vendor_id'].value_counts().sort_index())*100
```


![png](images/nyc/output_7_0.png)




<pre style="background-color:white"><code>1    46.504973
2    53.495027
Name: vendor_id, dtype: float64
</code></pre>



**Pickup datetime**


```python
train_df.resample('D',on='pickup_datetime').count()['id'].plot(style='o-')
plt.ylabel('Counts per day')
plt.xlabel('Pickup datetime')
plt.show()
```


![png](images/nyc/output_9_0.png)



```python
np.argmin(train_df.resample('D',on='pickup_datetime').count()['id'])
```




    Timestamp('2016-01-23 00:00:00', freq='D')




```python
test_df.resample('D',on='pickup_datetime').count()['id'].plot(style='o-',c='green')
plt.ylabel('Counts per day')
plt.xlabel('Pickup datetime')
plt.show()
```


![png](images/nyc/output_11_0.png)



**Pickup longitude	& pickup latitude**


```python
plt.figure(figsize=(15,10))
plt.scatter(train_df.pickup_longitude,train_df.pickup_latitude,s=0.3,alpha=0.3)
plt.xlim(-74.03, -73.75)
plt.ylim(40.63, 40.85)
plt.xlabel('Pickup longitude')
plt.ylabel('Pickup latitude')
plt.show()
```


![png](images/nyc/output_14_0.png)


**Dropoff longitude & dropoff latitude**


```python
plt.figure(figsize=(15,10))
plt.scatter(train_df.dropoff_longitude,train_df.dropoff_latitude,s=0.3,alpha=0.2)
plt.xlim(-74.03, -73.75)
plt.ylim(40.63, 40.85)
plt.xlabel('Dropoff longitude')
plt.ylabel('Dropoff latitude')
plt.show()
```


![png](images/nyc/output_16_0.png)


**Passenger count**


```python
train_df['passenger_count'].value_counts().sort_index().plot(kind='bar')
plt.ylabel('Frequency')
plt.xlabel('Number of passengers')
plt.show()
print(train_df['passenger_count'].value_counts().sort_index())
```


![png](images/nyc/output_18_0.png)


<pre style="background-color:white"><code>    0         60
    1    1033540
    2     210318
    3      59896
    4      28404
    5      78088
    6      48333
    7          3
    8          1
    9          1
    Name: passenger_count, dtype: int64
</code></pre>


**Store and forward flag**


```python
train_df['store_and_fwd_flag'].value_counts().sort_index().plot(kind='bar')
plt.xticks(rotation='horizontal')
plt.ylabel('Frequency')
plt.xlabel('Store and forward flag')
plt.show()
print(train_df['store_and_fwd_flag'].value_counts().sort_index()/sum(train_df['store_and_fwd_flag'].value_counts().sort_index())*100)
```


![png](images/nyc/output_20_0.png)


    N    99.44846
    Y     0.55154
    Name: store_and_fwd_flag, dtype: float64


** Trip duration**


```python
np.log(train_df['trip_duration']).plot(kind='hist',bins=200,figsize=(15,5))
plt.xlabel('log(trip duration)')
plt.show()
```


![png](images/nyc/output_22_0.png)



```python
print("Maximum trip duration (in hours): {}".format(max(train_df['trip_duration'])//3600))
```

    Maximum trip duration (in hours): 979



```python
df = train_df[train_df.trip_duration < np.exp(11)]
plt.hist(np.log(df.trip_duration),bins=100)
plt.xlabel('log(trip duration)')
plt.ylabel('Frequency')
plt.show()
```


![png](images/nyc/output_24_0.png)

