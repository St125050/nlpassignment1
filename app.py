import streamlit as st
import numpy as np
import pickle

# Load the pickled model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Example function to get embedding for a word using a loaded model (GloVe or similar)
def get_embed(model, word):
    # Assuming model is a word-to-embedding dictionary (for GloVe, for example)
    if word in model:
        return model[word]
    else:
        return np.zeros(300)  # Return a zero vector if word not found (300 is common dimension)

# Function to compute the dot product and retrieve top 10 most similar contexts
def compute_dot_product(query, model, corpus_embeddings):
    # Get the embedding for the query word
    query_embedding = get_embed(model, query)
    
    # Ensure query_embedding is a valid numpy array with the expected shape
    if query_embedding is None or query_embedding.shape != (300,):
        return []

    # Compute dot products for all embeddings in the corpus
    similarities = []
    for word, embedding in corpus_embeddings.items():
        if embedding is None or embedding.shape != (300,):
            continue  # Skip invalid embeddings
        similarity = np.dot(query_embedding, embedding)
        similarities.append((word, similarity))
    
    # Sort similarities and return the top 10 most similar contexts
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:10]

# Streamlit app to handle the user input and display results
def main():
    # Load the model (update the path to your actual model)
    model_path = 'embed_glove1.pkl'  # Update the path to your pickled model
    model = load_model(model_path)

    # For simplicity, assume we have a corpus of words already embedded
    corpus_embeddings = {word: get_embed(model, word) for word in model.keys()}  # Example corpus
    
    # Title of the app
    st.title("Search Similar Contexts")

    # Input box for query
    query = st.text_input("Enter a word or query:", "")

    # Display a message if no query is entered
    if query:
        top_10_similar = compute_dot_product(query, model, corpus_embeddings)
        
        # Display results
        if top_10_similar:
            st.write(f"Top 10 Similar Contexts for: **{query}**")
            
            # Show results as a table
            result_data = [{"Word": word, "Similarity": similarity} for word, similarity in top_10_similar]
            st.table(result_data)
        else:
            st.write("No similar contexts found.")
    else:
        st.write("Please enter a query to search for similar contexts.")

if __name__ == "__main__":
    main()
