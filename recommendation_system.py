import os
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class GeneralRecommendationSystem:
    def __init__(self, feature_column, id_column, name_column, max_features=5000, stop_words='english'):
        """
        Initializes the recommendation system with the desired feature, id, and name columns.

        Args:
        feature_column (str): The column name that contains the features to be vectorized (e.g., 'tags').
        id_column (str): The column name that contains the unique ID for each item (e.g., 'id').
        name_column (str): The column name that contains the name of each item (e.g., 'name').
        max_features (int, optional): The maximum number of features for the vectorizer. Defaults to 5000.
        stop_words (str, optional): Stop words for the vectorizer. Defaults to 'english'.
        """
        self.feature_column = feature_column
        self.id_column = id_column
        self.name_column = name_column.lower()
        self.vectorizer = HashingVectorizer(n_features=max_features, stop_words=stop_words)

    def update_data_and_similarity(self, new_data, file_prefix="default", allow_duplicates=False):
        """
        Updates the dataset with new data and recalculates the similarity matrix.

        Args:
        new_data (list of dicts): Each dict should contain relevant fields, including the `feature_column`, `id_column`, and `name_column`.
        file_prefix (str, optional): Prefix for saving data and similarity matrix files. Defaults to 'default'.
        allow_duplicates (bool, optional): If True, allows duplicate entries. Defaults to False (no duplicates allowed).
        
        Returns:
        None: Saves updated data and similarity matrix as pickle files.
        """
        # Ensure 'data' and 'artifacts' directories exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('artifacts', exist_ok=True)

        # Convert new_data to a dataframe
        new_df = pd.DataFrame(new_data)

        # Standardize data by trimming whitespace and converting to lowercase where applicable
        new_df[self.id_column] = new_df[self.id_column].astype(str).str.strip()  # Ensure IDs are strings without extra spaces
        new_df[self.name_column] = new_df[self.name_column].str.lower().str.strip()  # Normalize names to lowercase
        new_df[self.feature_column] = new_df[self.feature_column].str.lower().str.strip()  # Normalize tags to lowercase

        # Load existing data if available, else create an empty dataframe
        try:
            existing_data = pd.read_csv(f"data/{file_prefix}_data.csv")
            # Normalize existing data as well to ensure consistency in comparison
            existing_data[self.id_column] = existing_data[self.id_column].astype(str).str.strip()
            existing_data[self.name_column] = existing_data[self.name_column].str.lower().str.strip()
            existing_data[self.feature_column] = existing_data[self.feature_column].str.lower().str.strip()
        except FileNotFoundError:
            existing_data = pd.DataFrame(columns=new_df.columns)

        # Combine the existing and new data
        combined_data = pd.concat([existing_data, new_df], ignore_index=True)

        # Handle duplicates based on the allow_duplicates argument
        if not allow_duplicates:
            # Remove duplicates based on id, name, and tags after standardizing
            combined_data = combined_data.drop_duplicates(subset=[self.id_column, self.name_column, self.feature_column])

        # Save the combined data to CSV (unique data or with duplicates based on user input)
        combined_data.to_csv(f"data/{file_prefix}_data.csv", index=False)

        # Vectorize the feature column
        combined_vector = self.vectorizer.fit_transform(combined_data[self.feature_column])

        # Compute the cosine similarity
        similarity_matrix = cosine_similarity(combined_vector)

        # Save the updated data and similarity matrix
        pickle.dump(combined_data, open(f"artifacts/{file_prefix}_data.pkl", "wb"))
        pickle.dump(similarity_matrix, open(f"artifacts/{file_prefix}_similarity.pkl", "wb"))

        print(f"Data and similarity matrix updated and saved successfully. {'Duplicates allowed.' if allow_duplicates else 'Duplicates removed.'}")


    def recommend(self, selected_item_name, file_prefix="default", top_n=5, allow_duplicates=False):
        """
        Recommends unique items based on a selected item's name.

        Args:
        selected_item_name (str): The name of the selected item.
        file_prefix (str, optional): Prefix for loading data and similarity matrix files. Defaults to 'default'.
        top_n (int, optional): The number of unique recommendations to return. Defaults to 5.
        
        Returns:
        list: A list of dictionaries containing the recommended items' id and name.
        """
        # Load the updated data and similarity matrix
        data = pickle.load(open(f"artifacts/{file_prefix}_data.pkl", "rb"))
        similarity_matrix = pickle.load(open(f"artifacts/{file_prefix}_similarity.pkl", "rb"))
        selected_item_name = selected_item_name.lower() if selected_item_name else None
        # Find the index of the selected item by name
        if selected_item_name not in data[self.name_column].values:
            print(f"Item '{selected_item_name}' not found in the dataset.")
            return []

        index = data[data[self.name_column] == selected_item_name].index[0]

        # Get similarity scores and sort them
        distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])

        # Track unique recommendations and exclude duplicates
        recommended_items = []
        seen_items = set()  # Set to track already recommended items

        for i in distances:
            recommended_index = i[0]
            recommended_item = data.iloc[recommended_index]
            
            # Ensure the recommended item is not the input item and not already recommended
            if recommended_item[self.name_column] != selected_item_name :
                if not allow_duplicates and recommended_item[self.name_column] in seen_items:
                    continue
                recommended_items.append({
                    self.id_column: recommended_item[self.id_column],
                    self.name_column: recommended_item[self.name_column]
                })
                seen_items.add(recommended_item[self.name_column])  # Mark this item as recommended

            # Stop when we've collected the desired number of unique recommendations
            if len(recommended_items) >= top_n:
                break

        return recommended_items
