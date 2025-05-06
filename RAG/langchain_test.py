import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ["OPENAI_API_KEY"] = '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'

from openai import OpenAI
import openai
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

if __name__ == '__main__':
    embeddings_model = OpenAIEmbeddings()
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
        "At Educative, we think RAG is the future of AI!"
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

    # create qugment query

    # define template
    template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say 'thanks for asking!' at the end of the answer.

        {context}
        Question: {question}

        Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)
    print(custom_rag_prompt)

    question = "what is future of AI?"
    context = retriever.invoke(question)
    print(f'context:{context} \n question is {question}')


    augmented_query = custom_rag_prompt.format(context = context, question = question)
    print(f'augmented query is {augmented_query}')

    # generator
    llm = ChatOpenAI(model = 'gpt-4o')

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} # pass context and question
        | custom_rag_prompt # format prompt using custome RAG prompt template
        | llm # use llm to generate response
        | StrOutputParser() # parse output to a string
    )

    response = rag_chain.invoke("what is the future of AI?")
    print(f'response is {response}')

    response1 = rag_chain.invoke("what is the future of Mahesh?")
    print(f'response is {response1}')




