from flask import Flask, request, render_template
import pandas as pd
import random

app = Flask(__name__)

# Load files
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# Recommendations functions
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

def get_random_products(n):
    return train_data.sample(n)

# List of predefined image URLs
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]

@app.route("/")
def index():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price))

@app.route("/main")
def main():
    return render_template('main.html')

@app.route("/all-products")
def all_products():
    search_query = request.args.get('search')
    
    if search_query:
        filtered_products = train_data[train_data.apply(lambda row: search_query.lower() in row.astype(str).str.lower().values, axis=1)]
    else:
        filtered_products = get_random_products(20)  # Default 20 products
    
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(filtered_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    
    return render_template('all_products.html', all_products=filtered_products, truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price))

if __name__ == '__main__':
    app.run(debug=True)
