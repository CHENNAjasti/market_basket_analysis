import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from csv import reader
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
groceries = []
with open('C:/Users/chenn/OneDrive/Documents/HTML docs/intern/groceries.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        groceries.append(row)
encoder = TransactionEncoder()
transactions = encoder.fit(groceries).transform(groceries)
transactions = transactions.astype('int')
df = pd.DataFrame(transactions, columns=encoder.columns_)
frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)
print(frequent_itemsets.head())
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
frequent_itemsets[(frequent_itemsets['length'] == 1) & (frequent_itemsets['support'] >= 0.02) ][0:5]
frequent_itemsets[(frequent_itemsets['length'] == 2) & (frequent_itemsets['support'] >= 0.02)]    # Find all frequent itemsets of length 2 with minimum support of 2%.
rules = association_rules(frequent_itemsets, metric='support', min_threshold=0.02)   #top 10 association rules with minimum support of 2%
print(rules)