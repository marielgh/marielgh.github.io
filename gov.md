## Government Spending Contracts

The Government Spending Contracts dataset includes over 15 years of contracts, from 2000-2016, with the federal government. The data is publicly available at <https://www.usaspending.gov> and [here](https://public.enigma.com/browse/u-s-government-spending-contracts/1a932abc-8398-47ff-ad33-d1eb9a8958cc).
These contracts are mutually binding agreements between the federal government and another party (the seller) that obligates the seller to furnish supplies and services for the direct benefit of the government and obligates the government to pay for these services.

###  Contracts per agency

<style>
summary {
    font-weight: normal;
    font-size: 1em;
    font-family: 'adobe-clean','HelveticaNeue',Helvetica,Arial,sans-serif;
}

p {
    font-family: 'adobe-clean','HelveticaNeue',Helvetica,Arial,sans-serif;      
}

details {
    margin-bottom: 20px;    
}
</style>


<details>
<summary>Click to see the complete list of agencies</summary>
<pre style="background-color:white"><code>ind_agency,name_agency
1100, Executive Office of the President
1145, Peace Corps
1153, Trade and Development Agency
1200, Department of Agriculture
1300, Department of Commerce
1400, Department of the Interior 
1500, Department of Justice
1600, Department of Labor
1900, Department of State
2000, Department of the Treasury
2400, Office of Personnel Management
2700, Federal Communications Commission
2800, Social Security Administration
2900, Federal Trade Commission
3100, Nuclear Regulatory Commission
3300, Smithsonian Institution
3352, J. F. Kennedy Center for the Performing Arts
3355, National Gallery of Art
3400, International Trade Commission
3600, Department of Veterans Affairs
4100, Merit Systems Protection Board
4500, Equal Employment Opportunity Commission
4700, General Services Administration
4900, National Science Foundation
5000, Securities and Exchange Commission
5800, Federal Emergency Management Agency
5900, National Foundation on the Arts and the Humanities
5920, National Endowment for the Arts
5940, National Endowment for the Humanities
6000, Railroad Retirement Board
6100, Consumer Product Safety Commission
6300, National Labor Relations Board
6400, Tennessee Valley Authority
6500, Federal Maritime Commission
6800, Environmental Protection Agency
6900, Department of Transportation
7000, Department of Homeland Security 
7200, Agency for International Development
7300, Small Business Administration
7400, American Battle Monuments Commission
7500, Department of Health and Human Services
8000, National Aeronautics and Space Administration
8400, United States Soldiers' and Airmen's Home
84af, Armed forces Retirement Home
8600, Department of Housing and Urban Development
8800, National Archives and Records Administration
8900, Department of Energy
8961, Federal Energy Regulatory Commission
9000, Selective Service System
9100, Department of Education
9506, Federal Election Commission
9514, Occupational Safety and Health Review Commission
9516, Defense Nuclear Facilities Safety Board
9517, Commission on Civil Rights
9524, National Mediation Board
9531, United States Holocaust Memorial Museum
9568, Broadcasting Board of Governors
9577, Corporation for National and Community Service
9594, Court Services and offender Supervision Agency
9700, Department of Defense</code></pre>
</details>

**Percentage of contracts per agency**

<a href="images/gov/percentage_contracts_per_agency-100.png" ><img src="images/gov/percentage_contracts_per_agency-75.png"/></a>

<details>
<summary>Click to see the code that produced the plot above</summary>
<pre><code>dep = ['9700', '4700', '1500', '3600', '1400', '1200']
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
	
g_cat = pd.concat(list_g,axis=1)
g_cat = g_cat.fillna(0)
g_cat = g_cat.transpose()

import matplotlib.pyplot as plt
g_cat.plot.bar(figsize=(15,7),stacked=True)
ax = plt.subplot(111)
plt.ylabel('% of contracts')
chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.7, chartBox.height])
ax.legend(loc='right', bbox_to_anchor=(1.55, 0.5), shadow=False, ncol=1,prop={'size': 12},framealpha=0)
plt.savefig('percentage_contracts_per_agency.pdf', bbox_inches='tight')
plt.show()</code></pre>
</details>

###  Contracts per industry

<details>
<summary>Click to see the complete list of industries</summary>
<pre style="background-color:white"><code>naics code;industry
11; Agriculture, Forestry, Fishing and Hunting
21; Mining
22; Utilities
23; Construction
31-33; Manufacturing
42; Wholesale Trade
44-45; Retail Trade
45; Retail Trade
48-49; Transportation and Warehousing
51; Information
52; Finance and Insurance
53; Real Estate Rental and Leasing
54; Professional, Scientific, and Technical Services
55; Management of Companies and Enterprises
56; Administr., Support, Waste Mgmt. and Remediation Services
61; Educational Services
62; Health Care and Social Assistance
71; Arts, Entertainment, and Recreation
72; Accommodation and Food Services
81; Other Services (except Public Administration)
92; Public Administration</code></pre>
</details>

**Percentage of contracts per industry**

<a href="images/gov/percentage_contracts_per_industry-100.png" ><img src="images/gov/percentage_contracts_per_industry-70.png"/></a>

### Contracts for women and minorities

**Percentage of contracts for women and minorities owned businesses**

<img src="images/gov/percentage_minorities-100.png"/>

<summary>Click to see the code that produced the plot above</summary>
<pre><code>minorities = []
women = []
for df in list_df:
    minorities.append(df.minority_owned_business_flag.value_counts().loc['Y']/df.shape[0]*100)
    women.append(df.women_owned_flag.value_counts().loc['Y']/df.shape[0]*100)
	
years = np.arange(2000,2017)
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(8,6))
plt.plot(years,minorities,label='minority owned business')
plt.plot(years,women,label='women owned business')
plt.legend(loc='upper left', bbox_to_anchor=(0.6,0.3))
plt.xlim(2000,2016.5)
plt.xlabel('Year')
plt.ylabel('Percentage of contracts awarded')
plt.show()</code></pre>
</details>

###  Contracts per state

**Percentage of contracts per state over time**

<video src="videos/states.mp4" poster="videos/poster-states.png" style="max-width:100%" controls preload></video>
<br>
<details>
<summary>Click to see the code that produced the plots of the video</summary>
<pre><code>list_df2 = []
for df in list_df:
    df = df[(df.vendor_country_code=='USA: UNITED STATES OF AMERICA')|(df.vendor_country_code=='UNITED STATES')]\
    .groupby('vendor_state_code').vendor_country_code.count().reset_index()
    list_df2.append(df)	
		
for k,df in enumerate(list_df):
    df.columns = ['vendor_state_code','proportion']
    df['percentage'] = df['percentage']/(df['percentage'].sum())*100
        
    df = df[df['vendor_state_code'].isin(states)].copy()
    
    df['vendor_state_code'] = df['vendor_state_code'].apply(lambda x: x.upper())
    df['state'] = df['vendor_state_code'].apply(lambda x: states[x])

    df['text'] = df['state'].astype(str)


    data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = True,
        locations = df['vendor_state_code'],
        z = df['percentage'].astype(float),
        zmin=0,
        zmax=20,
        locationmode = 'USA-states',
        text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "% of contracts", thickness=40, len=0.8, titlefont =dict(size=15), tickfont =dict(size=15))
        ) ]

    layout = dict(
        title = 'Percentage of contracts per state in year '+str(2000+k),
        titlefont = dict(size=20),
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
    fig = dict( data=data, layout=layout)
    offline.iplot(fig, image='png',image_width=960,image_height=540,filename='states_map')
</code></pre>
</details>

### Seasonality of contracts

The dotted vertical lines correspond to the end of the business month on September in each year.
<a href="images/gov/timeseries-120.png" ><img src="images/gov/timeseries-85.png"/></a>


<details>
<summary>Click to see the code that produced the plot above</summary>
<pre><code>list_g = []
for df in list_df:
    df = df.set_index(pd.DatetimeIndex(df.signed_date))
    df = df.resample('BM').count().unique_transaction_id # BM = business month end frequency
    list_g.append(df)
ts = pd.concat(list_g).resample('BM').sum()

sept = ts[ts.index.map(lambda x: x.month) == 9].index.tolist()

plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots()
ts.plot(figsize=(12,6))
xposition = sept
for xc in xposition:
    plt.axvline(x=xc, color='grey', linestyle=':',alpha=0.4)
plt.xlim(pd.to_datetime('2001-01-01'),pd.to_datetime('2016-12-31'))
plt.xlabel('Date')
plt.yticks([100000,300000,500000])
ax.set_yticklabels(['100k','300k','500k'])
plt.ylabel('Number of contracts')
plt.savefig('timeseries.pdf', bbox_inches='tight')
plt.show()</code></pre>
</details>

