import openai
import numpy as np
from sklearn.neighbors import NearestNeighbors

def get_gpt4_embedding(text):
    response = openai.embeddings.create(
        input = text,
        model = 'text-embedding-3-small'
    )

    return response.data[0].embedding


if __name__ == '__main__':
    # add open ai  key here
    openai.api_key = ''
    # List of example documents to be used in the index
    documents = [
        "This is the Fundamentals of RAG course.",
        "Educative is an AI-powered online learning platform.",
        "There are several Generative AI courses available on Educative.",
        "I am writing this using my keyboard.",
        "JavaScript is a good programming language :)"
    ]

    #get embeddings for each document
    embeddings = [get_gpt4_embedding(text) for text in documents]
    embeddings = np.array(embeddings)
    print("--" * 50)
    print("This is how it looks after going through an embedding model:\n")
    print(embeddings)

    # Fit a NearestNeighbors model on the document embeddings using cosine similarity
    index = NearestNeighbors(n_neighbors=1, metric='cosine').fit(embeddings)


    # Function to query the index with a given text query
    def query_index(query):
        query_embedding = get_gpt4_embedding(query)
        query_embedding = np.array([query_embedding])
        distance, indices = index.kneighbors(query_embedding)
        return documents[indices[0][0]]


    # Example Query
    query = "What is JS?"
    print("Query:", query)
    result = query_index(query)  # Retrieve the most similar document to the query

    print("Retrieved document:", result)  # Print the retrieved document