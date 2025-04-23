import openai
from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings()

if __name__ == '__main__':
    openai.api_key = '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'

    # Generate embeddings for a list of documents
    embeddings = embeddings_model.embed_documents(
        [
        "This is the Fundamentals of RAG course.",
        "Educative is an AI-powered online learning platform.",
        "There are several Generative AI courses available on Educative.",
        "I am writing this using my keyboard.",
        "JavaScript is a good programming language"
        ]
    )
    print(f'len of embeddings is {len(embeddings)}')
    print(f'len of first embedding vector is {len(embeddings[0])}')

    # List of example documents to be used in the database
    documents = [
        "Python is a high-level programming language known for its readability and versatile libraries.",
        "Java is a popular programming language used for building enterprise-scale applications.",
        "JavaScript is essential for web development, enabling interactive web pages.",
        "Machine learning is a subset of artificial intelligence that involves training algorithms to make predictions.",
        "Deep learning, a subset of machine learning, utilizes neural networks to model complex patterns in data.",
        "The Eiffel Tower is a famous landmark in Paris, known for its architectural significance.",
        "The Louvre Museum in Paris is home to thousands of works of art, including the Mona Lisa.",
        "Artificial intelligence includes machine learning techniques that enable computers to learn from data.",
    ]

    #create vector db
    db = Chroma.from_texts(documents, OpenAIEmbeddings())
    print(db)

    #configure database as retriever
    retriever = db.as_retriever(
        search_type = 'similarity',
        search_kwargs = {'k':1}
    )

    result = retriever.invoke('Where can I find Mona Lisa?')
    print(result)
