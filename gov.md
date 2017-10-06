
[Dataset source](https://public.enigma.com/browse/u-s-government-spending-contracts/1a932abc-8398-47ff-ad33-d1eb9a8958cc)

<pre style="background-color:white"><code>ind_agency,name_agency
1100,EXECUTIVE OFFICE OF THE PRESIDENT
1145,PEACE CORPS
1153,TRADE AND DEVELOPMENT AGENCY
1200,Department of Agriculture
1300,Department of Commerce
1400,Department of the Interior 
1500,Department of Justice
1600,Department of Labor
1900,Department of State
2000,Department of the Treasury
2400,Office of Personnel Management
2700,FEDERAL COMMUNICATIONS COMMISSION
2800,SOCIAL SECURITY ADMINISTRATION
2900,FEDERAL TRADE COMMISSION
3100,NUCLEAR REGULATORY COMMISSION
3300,SMITHSONIAN INSTITUTION
3352,J. F. KENNEDY CENTER FOR THE PERFORMING ARTS
3355,NATIONAL GALLERY OF ART
3400,INTERNATIONAL TRADE COMMISSION
3600,Department of Veterans Affairs
4100,MERIT SYSTEMS PROTECTION BOARD
4500,EQUAL EMPLOYMENT OPPORTUNITY COMMISSION
4700,GENERAL SERVICES ADMINISTRATION
4900,NATIONAL SCIENCE FOUNDATION
5000,SECURITIES AND EXCHANGE COMMISSION
5800,FEDERAL EMERGENCY MANAGEMENT AGENCY
5900,National Foundation on the Arts and the Humanities
5920,NATIONAL ENDOWMENT FOR THE ARTS
5940,NATIONAL ENDOWMENT FOR THE HUMANITIES
6000,RAILROAD RETIREMENT BOARD
6100,CONSUMER PRODUCT SAFETY COMMISSION
6300,NATIONAL LABOR RELATIONS BOARD
6400,TENNESSEE VALLEY AUTHORITY
6500,FEDERAL MARITIME COMMISSION
6800,ENVIRONMENTAL PROTECTION AGENCY
6900,Department of Transportation
7000,DEPARTMENT OF HOMELAND SECURITY 
7200,AGENCY FOR INTERNATIONAL DEVELOPMENT
7300,SMALL BUSINESS ADMINISTRATION
7400,AMERICAN BATTLE MONUMENTS COMMISSION
7500,Department of Health and Human Services
8000,NATIONAL AERONAUTICS AND SPACE ADMINISTRATION
8400,United States Soldiers' and Airmen's Home
84AF,ARMED FORCES RETIREMENT HOME
8600,Department of Housing and Urban Development
8800,NATIONAL ARCHIVES AND RECORDS ADMINISTRATION
8900,Department of Energy
8961,FEDERAL ENERGY REGULATORY COMMISSION
9000,SELECTIVE SERVICE SYSTEM
9100,Department of Education
9506,FEDERAL ELECTION COMMISSION
9514,OCCUPATIONAL SAFETY AND HEALTH REVIEW COMMISSION
9516,DEFENSE NUCLEAR FACILITIES SAFETY BOARD
9517,COMMISSION ON CIVIL RIGHTS
9524,NATIONAL MEDIATION BOARD
9531,UNITED STATES HOLOCAUST MEMORIAL MUSEUM
9568,BROADCASTING BOARD OF GOVERNORS
9577,CORPORATION FOR NATIONAL AND COMMUNITY SERVICE
9594,COURT SERVICES AND OFFENDER SUPERVISION AGENCY
9700,Department of Defense
others,OTHERS</code></pre>


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
    list_g2.append(g_cat)
```


Plots: 

<a href="images/gov/percentage_contracts_per_agency-100.png" ><img src="images/gov/percentage_contracts_per_agency-75.png"/></a>


<a href="images/gov/percentage_budget_per_agency-100.png" ><img src="images/gov/percentage_budget_per_agency-75.png"/></a>


Video:

<video src="videos/states.mp4" poster="videos/poster-states.png" style="max-width:100%" controls preload></video>
