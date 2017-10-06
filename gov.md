## Government Spending Contracts

The Government Spending Contracts dataset includes over 15 years of contracts, from 2000-2016, with the federal government. The data is publicly available at <https://www.usaspending.gov> and [here](https://public.enigma.com/browse/u-s-government-spending-contracts/1a932abc-8398-47ff-ad33-d1eb9a8958cc).
These contracts are mutually binding agreements between the federal government and another party (the seller) that obligates the seller to furnish supplies and services for the direct benefit of the government and obligates the government to pay for these services.

###  Contracts per agency


hello

```python
dep = ['9700', '4700', '1500', '3600', '1400', '1200']
list_df = [df00,df01,df02,df03,df04,df05,df06,df07,df08,df09,df10,df11,df12,df13,df14,df15,df16]
list_g = []
for df in list_df:
    g_cat = df.groupby('maj_agency_cat').unique_transaction_id.count().reset_index()
    g_cat['ind_agency'] = g_cat.maj_agency_cat.apply(lambda x: x.split(':')[0])
    g_cat['name_agency'] = g_cat.maj_agency_cat.apply(lambda x: x.split(':')[1])
    g_cat = g_cat.groupby('ind_agency').unique_transaction_id.sum().sort_values(ascending=False).reset_index()
    g_cat['percentage'] = g_cat.unique_transaction_id/(g_cat.unique_transaction_id.sum())*100
    g_cat = g_cat[g_cat.ind_agency != ''].copy()
    g_cat = g_cat.drop('unique_transaction_id',axis=1)
    g_cat = g_cat.set_index('ind_agency')
    g_cat = g_cat.loc[dep]
    g_cat = g_cat.reset_index()
    g_cat['name_agency'] = g_cat['ind_agency'].apply(lambda x: dict_agency[x])
    g_cat = g_cat[['name_agency','percentage']].set_index('name_agency')
    list_g.append(g_cat)
```


**Percentage of contracts per agency**

<a href="images/gov/percentage_contracts_per_agency-100.png" ><img src="images/gov/percentage_contracts_per_agency-75.png"/></a>

**Percentage of budget per agency**

<a href="images/gov/percentage_budget_per_agency-100.png" ><img src="images/gov/percentage_budget_per_agency-75.png"/></a>

### Contracts for women and minorities

<img src="images/gov/percentage_minorities-100.png"/>

###  Contracts per state

**Percentage of contracts per state over time**

<video src="videos/states.mp4" poster="videos/poster-states.png" style="max-width:100%" controls preload></video>


