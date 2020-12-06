# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import pandas as pd


# %%
#######################################################################
#######################################################################
# Anaylsis of Census Data: 
#######################################################################
#######################################################################


# %%
# Reading census data 
df = pd.read_csv("/Users/mac/Desktop/census.csv")

# Selecting neighborhoods and proportions
new_df_n = df[['Neighborhood','White Proportion','Black Proportion']]

# Selecting sub- neighborhoods and proportions
new_df_sn = df[['Primary Sub-Neighborhood','White Proportion','Black Proportion']]

# Removing the % signs 
new_df_n['White Proportion'] = list(map(lambda x: x[:-1], new_df_n['White Proportion'].values))
new_df_n['Black Proportion'] = list(map(lambda x: x[:-1], new_df_n['Black Proportion'].values))
new_df_sn['White Proportion'] = list(map(lambda x: x[:-1], new_df_sn['White Proportion'].values))
new_df_sn['Black Proportion'] = list(map(lambda x: x[:-1], new_df_sn['Black Proportion'].values))

# Casting them as float
new_df_n['White Proportion'] = new_df_n['White Proportion'].astype(float)
new_df_n['Black Proportion'] = new_df_n['Black Proportion'].astype(float)
new_df_sn['White Proportion'] = new_df_sn['White Proportion'].astype(float)
new_df_sn['Black Proportion'] = new_df_sn['Black Proportion'].astype(float)

new_df_sn.dropna()

print(new_df_n)
new_df_n.info()

print(new_df_sn)
new_df_sn.info()


# %%
# Creating a pie chart to visualize the population proportion according to race
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt
# Pie chart
labels = ' Population of Black Americans ', 'Population of White Americans', 'Population of Other Races'
sizes = [(195752/664674),(389001/664674),(79921/664674)]
explode = (0.1, 0.1, 0.1)

fig1, ax1 = plt.subplots()
plt.title("Population Percentage According to Race")
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90,colors = [ "dodgerblue", "orange","green"])
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# %%
# Creating a list of the Predominant white neighborhoods
# ------------------------------------------------------
w_n = new_df_n.groupby('Neighborhood', axis= 0)['White Proportion'].mean()
w_n = w_n.to_frame()
b_n = new_df_n.groupby('Neighborhood', axis= 0)['Black Proportion'].mean()
b_n = b_n.to_frame()
c = w_n[w_n['White Proportion'] > b_n['Black Proportion']]
print(c.count())
w_neighborhoods = c.index.tolist()
w_neighborhoods


# %%
# Creating a list of the Predominant Black neighborhoods
# ------------------------------------------------------
b_n[b_n['Black Proportion']> w_n['White Proportion']]
b_n[b_n['Black Proportion']> w_n['White Proportion']].count()
d = b_n[b_n['Black Proportion']> w_n['White Proportion']]
print(d.count())
b_neighborhoods = d.index.tolist()
b_neighborhoods


# %%
import matplotlib.pyplot as plt
# Pie chart
labels = 'Predominant White Neighborhoods', 'Predominant Black Neighborhoods'
sizes = [(19/23),(4/23)]
explode = (0.1, 0.1)

fig1, ax1 = plt.subplots()
plt.title("Neighborhoods")
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90,colors = [ "dodgerblue", "orange","green"])

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# %%
# Creating a list of the Predominant white subneighborhoods
# ------------------------------------------------------
w_sn = new_df_sn.groupby('Primary Sub-Neighborhood', axis= 0)['White Proportion'].mean()
w_sn = w_sn.to_frame()
b_sn = new_df_sn.groupby('Primary Sub-Neighborhood', axis= 0)['Black Proportion'].mean()
b_sn = b_sn.to_frame()
a = w_sn[w_sn['White Proportion']>b_sn['Black Proportion']]
print(a.count())
# List of predomoinant white sub-neighborhoods
w_subneighborhoods = a.index.tolist()
w_subneighborhoods


# %%
# Creating a list of the Predominant black subneighborhoods
# ------------------------------------------------------
b_sn[b_sn['Black Proportion'] > w_sn['White Proportion']]
print(b_sn[b_sn['Black Proportion']>w_sn['White Proportion']].count())
b = b_sn[b_sn['Black Proportion'] > w_sn['White Proportion']]
# List of predomoinant black sub-neighborhoods
b_subneighborhoods = b.index.tolist()
b_subneighborhoods


# %%
import matplotlib.pyplot as plt
# Pie chart
labels = 'Predominant White Sub-Neighborhoods', 'Predominant Black Sub-Neighborhoods', 'Others'
sizes = [(61/87),(25/87),(1/87)]
explode = (0.1, 0.1, 0.1)  
fig1, ax1 = plt.subplots()
plt.title("Sub-Neighborhoods")
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90,colors = [ "dodgerblue", "orange","green"])


ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# %%
#######################################################################
#######################################################################
# WBUR and Census Data Analysis: 
#######################################################################
#######################################################################


# %%
# Working with the WBUR data set 
# Aggregating all the five years articles 

data1 = pd.read_csv("/Users/mac/Desktop/wbur2018.csv")
data2 = pd.read_csv("/Users/mac/Desktop/wbur2017.csv")
data3 = pd.read_csv("/Users/mac/Desktop/wbur2016.csv")
data4 = pd.read_csv("/Users/mac/Desktop/wbur2015.csv")
data5 = pd.read_csv("/Users/mac/Desktop/wbur2014.csv")

data_frame = pd.concat([data1,data2,data3,data4,data5], ignore_index=True)
data_frame


# %%
# Cleaning the data set 
# -----------------------

# Special characters 
spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–", "\xc2", "\xa0",
             "\x80", "\x9c", "\x99", "\x94", "\xad", "\xe2", "\x9d"]

# Removing Special characters 
for char in spec_chars:
    data_frame['text'] = data_frame['text'].str.replace(char, ' ')

# Lowercase all the words
data_frame['text'] = data_frame['text'].apply(lambda x: x.lower())


# %%
# Implementation on Neighborhoods 
# ----------------------------------------------------------


# %%
# Finding the white neighborhoods that were mentioned 
# ----------------------------------------------
data_arr = data_frame['text']
# -------------------------------------
wn_df = pd.DataFrame(columns =['Subs'])
wn_df['Subs'] = w_neighborhoods
# -------------------------------------

mentions=[]
for neighborhood in wn_df['Subs'].str.lower():
    count = 0
    for article in data_arr: 
        if neighborhood in article:
            count += 1
            continue
    mentions.append(count)

mention= pd.DataFrame(columns =['Count'])
mention['Count'] = mentions
print(mentions)
print(mention.sum())


# %%
# Creating a csv of white neighborhoods mentioned
# ------------------------------------------------------
new_df = pd.concat([wn_df,mention],axis=1)
new_df
selection = new_df.loc[new_df['Count']!=0]
selection.to_csv('white_neigh_mention.csv')


# %%
# Finding the Black neighborhoods that were mentioned 
# ----------------------------------------------
bn_df = pd.DataFrame(columns =['Subs'])
bn_df['Subs'] = b_neighborhoods
# ----------------------------------------------

mentions=[]
for neighborhood in bn_df['Subs'].str.lower():
    count = 0
    for article in data_arr: 
        if neighborhood in article:
            count += 1
            continue
    mentions.append(count)
    
mention= pd.DataFrame(columns =['Count'])
mention['Count'] = mentions
print(mentions)
print(mention.sum())


# %%
# Creating a csv of black neighborhoods mentioned
# ------------------------------------------------------
new_df = pd.concat([bn_df,mention],axis=1)
new_df
selection = new_df.loc[new_df['Count']!=0]
selection.to_csv('black_neigh_mention.csv')


# %%
# Implementation on Sub Neighborhoods 
# -------------------------------------
# white sub-neighborhoods mentioned
ws_df = pd.DataFrame(columns =['Subs'])
ws_df['Subs'] = w_subneighborhoods
# ------------------------------------

mentions=[]
for neighborhood in ws_df['Subs'].str.lower():
    count = 0
    for article in data_arr: 
        if neighborhood in article:
            count += 1
            continue 
    mentions.append(count)
    
mention= pd.DataFrame(columns =['Count'])
mention['Count'] = mentions
print(mentions)
print(mention.sum())


# %%
# Printing the list of white sub neighborhoods mentioned 
# ------------------------------------------------------
new_df = pd.concat([ws_df,mention],axis=1)
new_df
selection = new_df.loc[new_df['Count']!=0]
selection.to_csv('white_subneigh_mention.csv')


# %%
# Finding the Black neighborhoods that were mentioned 
# ---------------------------------------------------
s_df = pd.DataFrame(columns =['Subs'])
s_df['Subs'] = b_subneighborhoods
# -------------------------------

mentions=[]
for neighborhood in s_df['Subs'].str.lower():
    count = 0
    for article in data_arr: 
        if neighborhood in article:
            count += 1
    mentions.append(count)
    

mention= pd.DataFrame(columns =['Count'])
mention['Count'] = mentions
print(mention.sum())


# %%
# Printing the list of black sub neighborhoods mentioned 
# ------------------------------------------------------
new_df = pd.concat([s_df,mention],axis=1)
selection = new_df.loc[new_df['Count']!=0]
selection.to_csv('black_subneigh_mention.csv')


# %%
import matplotlib.pyplot as plt
# Pie chart
labels = 'Coverage of Black Sub-Neighborhoods', 'Coverage of White Sub-Neighborhoods', 'Others'
sizes = [(878/2671),(1673/2671),(120/2671)]
explode = (0.1, 0.1, 0.1)  

fig1, ax1 = plt.subplots()
plt.title("Sub-Neighborhood Coverage")
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90,colors = [ "orange","dodgerblue","green"])
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# %%
#######################################################################
#######################################################################
# WGBH and Census Data Analysis: 
#######################################################################
#######################################################################


# %%
data1 = pd.read_csv("/Users/mac/Desktop/wgbh_data_2018.csv")
data2 = pd.read_csv("/Users/mac/Desktop/wgbh_data_2017.csv")
data3 = pd.read_csv("/Users/mac/Desktop/wgbh_data_2016.csv")
data4 = pd.read_csv("/Users/mac/Desktop/wgbh_data_2015.csv")
data_frame = pd.concat([data1,data2,data3,data4], ignore_index=True)

data_frame


# %%
# Cleaning the data set 
data_frame['Article Content'] = data_frame['Article Content'].astype(str)
data_frame['Article Headline'] = data_frame['Article Headline'].astype(str)
# Special characters 
spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–", "\xc2", "\xa0",
             "\x80", "\x9c", "\x99", "\x94", "\xad", "\xe2", "\x9d"]

# Removing Special characters z
for char in spec_chars:
    data_frame['Article Content'] = data_frame['Article Content'].str.replace(char, ' ')

# Lowercase all the words
data_frame['Article Content'] = data_frame['Article Content'].apply(lambda x: x.lower())

data_frame


# %%
# ----------------------------------------------------------
# Implementation on Neighborhoods 
# ----------------------------------------------------------


# %%
# Finding the white neighborhoods that were mentioned 
# ----------------------------------------------
data_arr = data_frame['Article Content']
# -------------------------------------
wn_df = pd.DataFrame(columns =['Subs'])
wn_df['Subs'] = w_neighborhoods
# -------------------------------------

mentions=[]
for neighborhood in wn_df['Subs'].str.lower():
    count = 0
    for article in data_arr: 
        if neighborhood in article:
            count += 1
            continue
    mentions.append(count)

mention= pd.DataFrame(columns =['Count'])
mention['Count'] = mentions
print(mentions)
print(mention.sum())


# %%
# Printing the list of white neighborhoods mentioned 
# ------------------------------------------------------
new_df = pd.concat([wn_df,mention],axis=1)
new_df
selection = new_df.loc[new_df['Count']!=0]
selection


# %%
# This is done for the Black neighborhoods
# ----------------------------------------------
bn_df = pd.DataFrame(columns =['Subs'])
bn_df['Subs'] = b_neighborhoods
# ----------------------------------------------

mentions=[]
for neighborhood in bn_df['Subs'].str.lower():
    count = 0
    for article in data_arr: 
        if neighborhood in article:
            count += 1
            continue
    mentions.append(count)
    
mention= pd.DataFrame(columns =['Count'])
mention['Count'] = mentions
print(mentions)
print(mention.sum())


# %%
# Printing the list of black neighborhoods mentioned 
# ------------------------------------------------------
new_df = pd.concat([bn_df,mention],axis=1)
new_df
selection = new_df.loc[new_df['Count']!=0]
selection


# %%
# ----------------------------------------------------------
# Implementation on Sub Neighborhoods 
# ----------------------------------------------------------


# %%
# white sub-neighborhoods mentioned
ws_df = pd.DataFrame(columns =['Subs'])
ws_df['Subs'] = w_subneighborhoods
# ------------------------------------

mentions=[]
for neighborhood in ws_df['Subs'].str.lower():
    count = 0
    for article in data_arr: 
        if neighborhood in article:
            count += 1
            continue 
    mentions.append(count)
    
mention= pd.DataFrame(columns =['Count'])
mention['Count'] = mentions
print(mentions)
print(mention.sum())


# %%
# Printing the list of white sub neighborhoods mentioned 
# ------------------------------------------------------
new_df = pd.concat([ws_df,mention],axis=1)
new_df
selection = new_df.loc[new_df['Count']!=0]
selection


# %%
# Black sub-neighborhoods mentioned
s_df = pd.DataFrame(columns =['Subs'])
s_df['Subs'] = b_subneighborhoods
# -------------------------------

mentions=[]
for neighborhood in s_df['Subs'].str.lower():
    count = 0
    for article in data_arr: 
        if neighborhood in article:
            count += 1
    mentions.append(count)
    

mention= pd.DataFrame(columns =['Count'])
mention['Count'] = mentions
print(mention.sum())


# %%
# Printing the list of black sub neighborhoods mentioned 
# ------------------------------------------------------
new_df = pd.concat([s_df,mention],axis=1)
selection = new_df.loc[new_df['Count']!=0]
selection


# %%
#######################################################################
#######################################################################
# Homoicide Coverage 
#######################################################################
#######################################################################


# %%
homoicide_df = pd.read_csv('Homicide List Boston 2014-2018.csv')
homoicide_df.head()


# %%
a = homoicide_df.groupby("Year")
homoicide_df.info()


# %%
f_name = homoicide_df['First Name']
l_name = homoicide_df['Last Name']
f_name = f_name.str.lower()
l_name = l_name.str.lower()

f_df = pd.DataFrame(columns =['First Name'])
l_df = pd.DataFrame(columns =['Last Name'])
f_df['First Name'] = f_name
l_df['Last Name'] = l_name


# %%
mentions=[]
for i,j in zip(f_df['First Name'],l_df['Last Name']):
    count = 0
    for article in data_arr: 
        if i in article.split() and j in article.split():
            count += 1
    mentions.append(count)
    

mention= pd.DataFrame(columns =['Count'])
mention['Count'] = mentions
print(mention.sum())


# %%
new_df = pd.concat([homoicide_df,mention],axis=1)
selection = new_df.loc[new_df['Count']!=0]
selection


# %%
new_df.pivot_table(index='Race',values='Count', aggfunc=np.sum)


# %%
new_df.pivot_table(index='Gender',values='Count', aggfunc=np.sum)


# %%
new_df.pivot_table(index='Age',values='Count', aggfunc=np.sum)

