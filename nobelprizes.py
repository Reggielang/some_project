#  CMT114 Coursework

# student number:1961145

import pandas as pd
import matplotlib.pyplot as plt

def report(nobelprizeDict):
    # your code here
    # Open and read the file as a dataframe
    import json as js
    with open(nobelprizeDict) as data_l:
        data_l = data_l.read()
        data_l = js.loads(data_l)
        pd.set_option('display.max_columns', None)
        df = pd.DataFrame(data=data_l)
    # solution for find award_or_not:
        # First, observe the json file, some overallMotivation is No Price in the year of no award.
        # Find some rows  without winners by traversing the data set
    for laureates in df.columns:
        if df[laureates].count() != len(df):
            loc = df[laureates][df[laureates].isnull().values == True].index.tolist()
            # print('columnsname："{}", No.{} have missing'.format(laureates,loc))

    # Determine whether to win
    award_or_not = []
    for dic in data_l:
        if 'laureates' not in dic.keys():
            award_or_not.append(False)
        else:
            award_or_not.append(True)

    df['award_or_not'] = award_or_not

    df = pd.DataFrame(data=df, columns=['year', 'category', 'award_or_not'])
    df['year'] = df['year'].astype(int)
    df['category'] = df['category'].apply(str)
    return df
    # df.to_csv("award_data.csv",columns=['year','category','award_or_not'],index=True,header=True)

def get_laureates_and_motivation(nobelprizeDict, year, category):
    # your code here
    # Open and read the file as a dataframe
    import json as js
    import numpy as np
    with open(nobelprizeDict) as data_l:
        data_l = data_l.read()
        data_l = js.loads(data_l)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        df = pd.DataFrame(data=data_l, columns=['year', 'category', 'laureates', 'overallMotivation'])

    df_0 = df.dropna(subset=['laureates'])

    # Correspond to the winners and award-winning years and awards
    df_year = pd.DataFrame(
        {'year': df_0.year.repeat(df_0.laureates.str.len()),
         'laureates': np.concatenate(df_0.laureates.values)})
    df_year = df_year.drop(['laureates'], axis=1)

    df_category = pd.DataFrame(
        {'category': df_0.category.repeat(df_0.laureates.str.len()),
         'laureates': np.concatenate(df_0.laureates.values)})
    df_category = df_category.drop(['laureates'], axis=1)

    df_overall = pd.DataFrame({'overallMotivation': df_0.overallMotivation.repeat(df_0.laureates.str.len()),
                               'laureates': np.concatenate(df_0.laureates.values)})
    df_overall = df_overall.drop(['laureates'], axis=1)
    df_overall = df_overall.fillna(np.nan)
    df_overall = df_overall.rename(columns={'overallMotivation': 'overall_motivation'})

    df_1 = pd.concat([df_category, df_year, df_overall], axis=1)
    df_1 = df_1.reset_index(drop=True)

    # Integrate the winner's name
    people = []
    for array in df['laureates'].values:
        if isinstance(array, list):
            for dic in array:
                people.append(dic)

    df_people = pd.DataFrame(data=people)
    df_people = df_people.drop('share', 1)
    df_people['laureate'] = df_people['firstname'].map(str) + ' ' + df_people['surname'].map(str)
    df_people = df_people.drop(['firstname', 'surname'], axis=1)
    df_2 = pd.DataFrame(data=df_people, columns=['id', 'laureate', 'motivation'])
    df = pd.concat([df_2, df_1], axis=1)
    order = ['category', 'year', 'id', 'laureate', 'motivation', 'overall_motivation']
    df = df[order]
    df[['year', 'id']] = df[['year', 'id']].astype(int)
    df[['category', 'laureate', 'motivation']] = df[['category', 'laureate', 'motivation']].astype(str)
    df = df[(df['year'] == year) & (df['category'] == category)]
    return df

def plot_freqs(nobelprizeDict):
    # your code here
    # The same processing from question 2.2
    import json as js
    import numpy as np
    with open(nobelprizeDict) as data_l:
        data_l = data_l.read()
        data_l = js.loads(data_l)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        df = pd.DataFrame(data=data_l, columns=['year', 'category', 'laureates', 'overallMotivation'])

    df_0 = df.dropna(subset=['laureates'])
    df_1 = pd.DataFrame(
        {'category': df_0.category,
         'year': df_0.year,
         'overallMotivation': df_0.overallMotivation.repeat(df_0.laureates.str.len()),
         'laureates': np.concatenate(df_0.laureates.values)})
    df_1 = df_1.rename(columns={'overallMotivation': 'overall_motivation'})
    df_1 = df_1.fillna(np.nan)
    df_1 = df_1.drop(['laureates'], axis=1)
    df_1 = df_1.reset_index(drop=True)

    people = []
    for array in df['laureates'].values:
        if isinstance(array, list):
            for dic in array:
                people.append(dic)

    df_people = pd.DataFrame(data=people)
    df_people = df_people.drop('share', 1)
    df_people['laureate'] = df_people['firstname'].map(str) + ' ' + df_people['surname'].map(str)
    df_people = df_people.drop(['firstname', 'surname'], axis=1)
    df_2 = pd.DataFrame(data=df_people, columns=['id', 'laureate', 'motivation'])
    df = pd.concat([df_2, df_1], axis=1)
    df[['year', 'id']] = df[['year', 'id']].astype(int)
    df[['category', 'laureate', 'motivation']] = df[['category', 'laureate', 'motivation']].astype(str)
    df = pd.DataFrame(data=df, columns=['category', 'motivation', 'overall_motivation'])
    # Separate treatment of six awards
    listType = df['category'].unique()
    data_che = df[df['category'].isin([listType[0]])]
    data_eco = df[df['category'].isin([listType[1]])]
    data_lit = df[df['category'].isin([listType[2]])]
    data_pea = df[df['category'].isin([listType[3]])]
    data_phy = df[df['category'].isin([listType[4]])]
    data_med = df[df['category'].isin([listType[5]])]

    with open('whitelist.txt') as word_data:
        words = list(word_data)
        words_l = list(map(lambda x: x.strip(), words))

    l_data_che = data_che['motivation'].values.tolist()
    l_data_che_overall = data_che['overall_motivation'].values.tolist()

    # For the chemical award word frequency
    counts_che = {}
    for word in words_l:
        counts_che[word] = 0

    for i in l_data_che:
        for word in words_l:
            if word in i.split():
                counts_che[word] += 1

    counts_che = {key: val for key, val in counts_che.items() if val != 0}
    counts_che_overall = {}
    for word in words_l:
        counts_che_overall[word] = 0

    for i in l_data_che_overall:
        if i is not np.nan:
            for word in words_l:
                if word in i.split():
                    counts_che_overall[word] += 1
    # Integrate all word frequencies ( motivation and overall)
    counts_che_overall = {key: val for key, val in counts_che_overall.items() if val != 0}
    for k, v in counts_che_overall.items():
        if k in counts_che.keys():
            counts_che[k] += v
        else:
            counts_che[k] = v

    # For the economic award word frequency
    l_data_eco = data_eco['motivation'].values.tolist()
    l_data_eco_overall = data_eco['overall_motivation'].values.tolist()
    counts_eco = {}
    for word in words_l:
        counts_eco[word] = 0

    for i in l_data_eco:
        for word in words_l:
            if word in i.split():
                counts_eco[word] += 1

    counts_eco = {key: val for key, val in counts_eco.items() if val != 0}

    counts_eco_overall = {}
    for word in words_l:
        counts_eco_overall[word] = 0

    for i in l_data_eco_overall:
        if i is not np.nan:
            for word in words_l:
                if word in i.split():
                    counts_eco_overall[word] += 1
    # Integrate all word frequencies ( motivation and overall)
    counts_eco_overall = {key: val for key, val in counts_eco_overall.items() if val != 0}
    for k, v in counts_eco_overall.items():
        if k in counts_eco.keys():
            counts_eco[k] += v
        else:
            counts_eco[k] = v

    # For the literary award word frequency
    l_data_lit = data_lit['motivation'].values.tolist()
    l_data_lit_overall = data_lit['overall_motivation'].values.tolist()
    counts_lit = {}
    for word in words_l:
        counts_lit[word] = 0

    for i in l_data_lit:
        for word in words_l:
            if word in i.split():
                counts_lit[word] += 1

    counts_lit = {key: val for key, val in counts_lit.items() if val != 0}

    counts_lit_overall = {}
    for word in words_l:
        counts_lit_overall[word] = 0

    for i in l_data_lit_overall:
        if i is not np.nan:
            for word in words_l:
                if word in i.split():
                    counts_lit_overall[word] += 1
    # Integrate all word frequencies ( motivation and overall)
    counts_lit_overall = {key: val for key, val in counts_lit_overall.items() if val != 0}
    for k, v in counts_lit_overall.items():
        if k in counts_lit.keys():
            counts_lit[k] += v
        else:
            counts_lit[k] = v

    # For the peace award word frequency
    l_data_pea = data_pea['motivation'].values.tolist()
    l_data_pea_overall = data_pea['overall_motivation'].values.tolist()
    counts_pea = {}
    for word in words_l:
        counts_pea[word] = 0

    for i in l_data_pea:
        for word in words_l:
            if word in i.split():
                counts_pea[word] += 1

    counts_pea = {key: val for key, val in counts_pea.items() if val != 0}

    counts_pea_overall = {}
    for word in words_l:
        counts_pea_overall[word] = 0

    for i in l_data_pea_overall:
        if i is not np.nan:
            for word in words_l:
                if word in i.split():
                    counts_pea_overall[word] += 1
    # Integrate all word frequencies ( motivation and overall)
    counts_pea_overall = {key: val for key, val in counts_pea_overall.items() if val != 0}
    for k, v in counts_pea_overall.items():
        if k in counts_pea.keys():
            counts_pea[k] += v
        else:
            counts_pea[k] = v

    # For the physics award word frequency
    l_data_phy = data_phy['motivation'].values.tolist()
    l_data_phy_overall = data_phy['overall_motivation'].values.tolist()
    counts_phy = {}
    for word in words_l:
        counts_phy[word] = 0

    for i in l_data_phy:
        for word in words_l:
            if word in i.split():
                counts_phy[word] += 1

    counts_phy = {key: val for key, val in counts_phy.items() if val != 0}

    counts_phy_overall = {}
    for word in words_l:
        counts_phy_overall[word] = 0

    for i in l_data_phy_overall:
        if i is not np.nan:
            for word in words_l:
                if word in i.split():
                    counts_phy_overall[word] += 1
    # Integrate all word frequencies ( motivation and overall)
    counts_phy_overall = {key: val for key, val in counts_phy_overall.items() if val != 0}
    for k, v in counts_phy_overall.items():
        if k in counts_phy.keys():
            counts_phy[k] += v
        else:
            counts_phy[k] = v

    # For the medicine award word frequency
    l_data_med = data_med['motivation'].values.tolist()
    l_data_med_overall = data_med['overall_motivation'].values.tolist()
    counts_med = {}
    for word in words_l:
        counts_med[word] = 0

    for i in l_data_med:
        for word in words_l:
            if word in i.split():
                counts_med[word] += 1

    counts_med = {key: val for key, val in counts_med.items() if val != 0}
    counts_med_overall = {}
    for word in words_l:
        counts_med_overall[word] = 0

    for i in l_data_med_overall:
        if i is not np.nan:
            for word in words_l:
                if word in i.split():
                    counts_med_overall[word] += 1
    # Integrate all word frequencies ( motivation and overall)
    counts_med_overall = {key: val for key, val in counts_med_overall.items() if val != 0}
    for k, v in counts_med_overall.items():
        if k in counts_med.keys():
            counts_med[k] += v
        else:
            counts_med[k] = v
    # Sort the dictionary and convert them to dataframe
    counts_che = sorted(counts_che.items(), key=lambda x: x[1], reverse=True)
    counts_eco = sorted(counts_eco.items(), key=lambda x: x[1], reverse=True)
    counts_lit = sorted(counts_lit.items(), key=lambda x: x[1], reverse=True)
    counts_pea = sorted(counts_pea.items(), key=lambda x: x[1], reverse=True)
    counts_phy = sorted(counts_phy.items(), key=lambda x: x[1], reverse=True)
    counts_med = sorted(counts_med.items(), key=lambda x: x[1], reverse=True)

    counts_che = pd.DataFrame(data=counts_che, columns=['words', 'frequency'])
    counts_eco = pd.DataFrame(data=counts_eco, columns=['words', 'frequency'])
    counts_lit = pd.DataFrame(data=counts_lit, columns=['words', 'frequency'])
    counts_pea = pd.DataFrame(data=counts_pea, columns=['words', 'frequency'])
    counts_phy = pd.DataFrame(data=counts_phy, columns=['words', 'frequency'])
    counts_med = pd.DataFrame(data=counts_med, columns=['words', 'frequency'])

    words_che = counts_che.loc[[0, 9, 19, 29, 39, 49], :]
    words_eco = counts_eco.loc[[0, 9, 19, 29, 39, 49], :]
    words_lit = counts_lit.loc[[0, 9, 19, 29, 39, 49], :]
    words_pea = counts_pea.loc[[0, 9, 19, 29, 39, 49], :]
    words_phy = counts_phy.loc[[0, 9, 19, 29, 39, 49], :]
    words_med = counts_med.loc[[0, 9, 19, 29, 39, 49], :]
    # show data with matplotlib
    words_map = {"chemistry": words_che, 'economics': words_eco,
                 'literature': words_lit, 'peace': words_pea, 'physics': words_phy,
                 'medicine': words_med}
    for ele in words_map.keys():
        words = words_map[ele]
        x = words['words']
        y = words['frequency']
        plt.scatter(x, y, s=10 * np.array(y), c=y)
        plt.title(ele, fontsize=16)
        plt.show()