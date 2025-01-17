import pickle
import streamlit as st
import numpy as np

# Define function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 != 0 and norm_vec2 != 0 else 0.0

# Load the embedding dictionaries
def load_embeddings():
    pickle_file_path_pos = 'embed_skipgram_positive.pkl'
    pickle_file_path_neg = 'embed_skipgram_negative.pkl'
    pickle_file_path_glove = 'embed_glove.pkl'
    
    with open(pickle_file_path_pos, 'rb') as file:
        embedding_dict_pos = pickle.load(file)
    with open(pickle_file_path_neg, 'rb') as file:
        embedding_dict_neg = pickle.load(file)
    with open(pickle_file_path_glove, 'rb') as file:
        embedding_dict_glove = pickle.load(file)
    
    return embedding_dict_pos, embedding_dict_neg, embedding_dict_glove

# Function to find next 10 similar words based on cosine similarity
def find_next_10_cosine_words_for_word(target_word, embedding_dict, top_n=10):
    if target_word not in embedding_dict:
        return ["Word not in Corpus"]
    
    target_vector = embedding_dict[target_word]
    similarities = []
    
    for word, vector in embedding_dict.items():
        if word != target_word:  # Skip the target word itself
            sim = cosine_similarity(target_vector, vector)
            similarities.append((word, sim))
    
    # Sort the words based on similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top N similar words
    return [word for word, _ in similarities[:top_n]]

# Streamlit app interface
def main():
    # Load embeddings at the start
    embedding_dict_pos, embedding_dict_neg, embedding_dict_glove = load_embeddings()

    # Set up the Streamlit app title and description
    st.title("Word Similarity Search")
    st.write("Enter a word to find the next 10 most similar words based on cosine similarity.")

    # Get user input for the target word
    user_target_word = st.text_input("Enter a word to search:", "run")  # Default word is "run"

    # Select the embedding model to use
    model_choice = st.selectbox("Select Embedding Model", ["GloVe", "Skipgram Positive", "Skipgram Negative"])

    # Based on the model choice, select the appropriate embedding dictionary
    if model_choice == "GloVe":
        embedding_dict = embedding_dict_glove
    elif model_choice == "Skipgram Positive":
        embedding_dict = embedding_dict_pos
    elif model_choice == "Skipgram Negative":
        embedding_dict = embedding_dict_neg

    # If the user entered a word, compute the top 10 similar words
    if user_target_word:
        with st.spinner('Finding similar words...'):
            next_10_cosine_for_user_word = find_next_10_cosine_words_for_word(user_target_word, embedding_dict, top_n=10)

            # Display results
            if next_10_cosine_for_user_word == ["Word not in Corpus"]:
                st.error("Word not in Corpus")
            else:
                st.success(f"Top 10 similar words for '{user_target_word}':")
                st.write(next_10_cosine_for_user_word)

if __name__ == "__main__":
    main()
