
# General Recommendation System

  

This project provides a general-purpose recommendation system that can be used for any type of data (e.g., products, movies, etc.). It calculates similarities between items based on specified features and provides recommendations using cosine similarity.

  

## Features

  

-  **Flexible Data Input**: You can import any list of dictionaries representing data for items (e.g., products, movies) and get recommendations based on specified features (tags, attributes).

-  **Duplicate Handling**: The system allows you to control whether to allow or disallow duplicate data when updating the dataset.

-  **Cosine Similarity**: The system uses cosine similarity to calculate recommendations based on the feature vectorization using `HashingVectorizer`.

-  **Customizable**: You can easily configure the columns for `id`, `name`, and `tags` to adapt it to different datasets.

  

## How It Works

  

The recommendation system works in the following way:

1.  **Data Handling**: Data is loaded from a list of dictionaries where each dictionary represents an item.

2.  **Vectorization**: The feature column (e.g., `tags`) is vectorized using a `HashingVectorizer` to convert text into feature vectors.

3.  **Cosine Similarity**: Cosine similarity is calculated between the feature vectors to determine the similarity between items.

4.  **Duplicate Handling**: If `allow_duplicates` is set to `False`, duplicates are removed based on the `id`, `name`, and `tags`.

  

## Setup and Installation

  

To get started with the recommendation system, follow these steps to set up the environment:

  

### 1. Clone the Repository

  

First, clone the repository from GitHub to your local machine:

  

```bash
git  clone  https://github.com/Mussu221/general_recommendation.git

cd  recommendation-system
```

2.  Create  a  Virtual  Environment

It  is  recommended  to  use  a  virtual  environment  to  manage  dependencies.  To  create  a  virtual  environment:

  

For  Windows:
```bash
python  -m  venv  venv
```
For  macOS/Linux:
```bash
python3  -m  venv  venv
```
  

3.  Activate  the  Virtual  Environment

Activate  the  virtual  environment  to  isolate  your  project's dependencies:

For Windows:
```bash
.\venv\Scripts\activate
```

For macOS/Linux:
```bash
source venv/bin/activate
```
  

4. Install Required Packages

Install the necessary packages for the recommendation system using requirements.txt:

  ```bash
pip install -r requirements.txt
```
  
  

This will install the following dependencies:

pandas
scikit-learn
numpy


5. Running the Program

Once you have set up the environment and installed the required packages, you can use the recommendation system in your projects.

  

# Example usage:
```python
from general_recommendation_system import GeneralRecommendationSystem
```
  

## Sample dataset
```python
custom_data = [

{"id": 101, "name": "Apple iPhone 13", "tags": "electronics smartphone mobile phone iOS A15 Bionic camera display"},

{"id": 102, "name": "Samsung Galaxy S21", "tags": "electronics smartphone mobile phone Android Snapdragon 888 AMOLED"},
]
```
  

# Initialize the recommendation system
```python
rec_system = GeneralRecommendationSystem(feature_column="tags", 
				id_column="id", name_column="name")
```
  

# Update dataset and calculate similarity 
```python
rec_system.update_data_and_similarity(custom_data,
		 file_prefix="product_recommendations", allow_duplicates=False)
```
  

# Recommend based on a selected product
```python
recommendations = rec_system.recommend("Apple iPhone 13",
					file_prefix="product_recommendations", top_n=5)

print(recommendations)
```