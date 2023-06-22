import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
users = pd.read_csv('users.csv')
products = pd.read_csv('products.csv')
ratings = pd.read_csv('ratings.csv')

# Merge data
user_ratings = pd.merge(ratings, users, on='user_id')
data = pd.merge(user_ratings, products, on='product_id')

# Create user-product matrix
matrix = data.pivot_table(index='user_id', columns='product_id', values='rating')

# Fill missing values with 0
matrix.fillna(0, inplace=True)

# Calculate similarity matrix
similarity = cosine_similarity(matrix)

# Define function to get top n recommendations for a user based on user demographics
def get_demographic_recommendations(user_demographics, n):
    # Get users with similar demographics
    similar_users = users[(users['age'] == user_demographics['age']) & (users['gender'] == user_demographics['gender']) & (users['income'] == user_demographics['income'])]

    # Get products rated by similar users
    top_products = matrix.loc[similar_users.index].sum().sort_values(ascending=False)

    # Get top n recommendations
    recommendations = top_products.head(n)

    # Convert product_id to product_name
    product_names = []
    for product_id in recommendations.index:
        product_name = products.loc[products['product_id'] == product_id]['name'].iloc[0]
        product_names.append(product_name)

    return product_names
# Get user demographics
age = input("Enter your age: ")
gender = input("Enter your gender: ")
income = input("Enter your income level: ")

# Create user demographics dictionary
user_demographics = {'age': int(age), 'gender': gender, 'income': income}

n = 5 # maximum number of recommended products
recommendations = get_demographic_recommendations(user_demographics, n)

# Print recommended products
print("Recommended products based on demographics :")
for product_name in recommendations:
    print(product_name)
