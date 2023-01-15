# import libraries
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load the data
data = [['milk', 'bread', 'butter'],
        ['milk', 'bread', 'butter', 'eggs'],
        ['milk', 'bread', 'banana'],
        ['milk', 'coffee', 'sugar', 'cornflakes'],
        ['coffee', 'sugar', 'cornflakes', 'bread'],
        ['coffee', 'sugar', 'cornflakes', 'bread', 'butter']]

# Transform the data into a binary
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Find the frequent item sets
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

# association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

print(rules)
