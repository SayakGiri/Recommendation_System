import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#importing the datasets

ds = pd.read_csv("E:\Online Courses, Internship and more\Recommendation system\posts.csv")
dataset = pd.read_csv("E:\Online Courses, Internship and more\Recommendation system\views.csv")
df = pd.read_csv("E:\Online Courses, Internship and more\Recommendation system\users.csv")

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(dataset['category'])

\tfidf_matrix = tf.fit_transform(ds['category'])


cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

results = {}

for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['_id'][i]) for i in similar_indices]

    results[row['_id']] = similar_items[1:]
    
print('done!')

def item(id):
    return ds.loc[ds['_id'] == id]['description'].tolist()[0].split(' - ')[0]

# Just reads the results out of the dictionary.
def recommend(item_id, num):
    print("Recommending " + str(num) + " posts similar to " + item(item_id) + "...")
    print("-------")
    recs = results[item_id][:num]
    for rec in recs:
        print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")

recommend(item_id=11, num=5)