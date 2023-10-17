# Can cause error......so execute the below command
# !pip install scikit-surprise

import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Define the Reader with the rating scale
reader=Reader(line_format='user item rating', rating_scale=(1, 5))

df = pd.read_csv("/fashion_products.csv")
print(df.head())

content_df = df[['Product ID', 'Product Name', 'Brand',
                   'Category', 'Color', 'Size']]
content_df['Content'] = content_df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

# Use TF-IDF vectorizer to convert content into a matrix of TF-IDF features
tfidf_vectorizer = TfidfVectorizer()
content_matrix = tfidf_vectorizer.fit_transform(content_df['Content'])

content_similarity = linear_kernel(content_matrix, content_matrix)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['User ID',
                                  'Product ID',
                                  'Rating']], reader)

def get_content_based_recommendations(product_id, top_n):
    index = content_df[content_df['Product ID'] == product_id].index[0]
    similarity_scores = content_similarity[index]
    similar_indices = similarity_scores.argsort()[::-1][1:top_n + 1]
    recommendations = content_df.loc[similar_indices, 'Product ID'].values
    return recommendations

# Load the DataFrame into a surprise Dataset
data = Dataset.load_from_df(df[['User ID', 'Product ID', 'Price']], reader)

# Build the full trainset
trainset = data.build_full_trainset()

algo = SVD()
algo.fit(trainset)

def get_collaborative_filtering_recommendations(user_id, top_n):
    testset = trainset.build_anti_testset()
    testset = filter(lambda x: x[0] == user_id, testset)
    predictions = algo.test(testset)
    predictions.sort(key=lambda x: x.est, reverse=True)
    recommendations = [prediction.iid for prediction in predictions[:top_n]]
    return recommendations

recommended_items = get_collaborative_filtering_recommendations(user_id=1, top_n=10)

def get_hybrid_recommendations(user_id, product_id, top_n):
    content_based_recommendations = get_content_based_recommendations(product_id, top_n)
    collaborative_filtering_recommendations = get_collaborative_filtering_recommendations(user_id, top_n)
    hybrid_recommendations = list(set(content_based_recommendations + collaborative_filtering_recommendations))
    return hybrid_recommendations[:top_n]

user_id = 6
product_id = 11
top_n = 10
recommendations = get_hybrid_recommendations(user_id, product_id, top_n)

print(f"Hybrid Recommendations for User {user_id} based on Product {product_id}:")
for i, recommendation in enumerate(recommendations):
    print(f"{i + 1}. Product ID: {recommendation}")
    print(f"{i + 1}. Product ID: {recommendation}")
