import pandas as pd
import numpy as np
from itertools import combinations
from operator import itemgetter
from time import time
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('t1.csv', usecols=['items'])

def my_aprior(data, support_count):
   
    start = time()
   
    for index, row in data.iterrows():
        data.loc[index, 'items'] = row['items'].strip()
   ems'].str.split(" ", expand = True)).apply(pd.value_counts) \
    .sum(axis = 1).where(lambda value: value > support_count).dropna()
    print("Find all frequent items")
   
    apriori_data = pd.DataFrame({'items': single_items.index.astype(int), 'support_count': single_items.values, 'set_size': 1})
    
    data['set_size'] = data['items'].str.count(" ") + 1
    data['items'] = data['items'].apply(lambda row: set(map(int, row.split(" "))))
    single_items_set = set(single_items.index.astype(int))
    
    for length in range(2, len(single_items_set) + 1):
        data = data[data['set_size'] >= length]
        d = data['items'] \
            .apply(lambda st: pd.Series(s if set(s).issubset(st) else None for s in combinations(single_items_set, length))) \
            .apply(lambda col: [col.dropna().unique()[0], col.count()] if col.count() >= support_count else None).dropna()
        if d.empty:
            break
        apriori_data = apriori_data.append(pd.DataFrame(
            {'items': list(map(itemgetter(0), d.values)), 'support_count': list(map(itemgetter(1), d.values)),
             'set_size': length}), ignore_index=True)
    print("End search, total time %s"%(time() - start))
    return apriori_data
apriori_result = my_aprior(dataset, 5000)
for x in apriori_result:
    print(x)
