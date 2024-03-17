from langid.langid import LanguageIdentifier, model  # type: ignore
from langchain import PromptTemplate  # type: ignore
from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
from langchain.vectorstores import FAISS  # type: ignore
from langchain.llms import CTransformers  # type: ignore
from langchain.chains import RetrievalQA # type: ignore
from langchain.chains.conversation.memory import ConversationBufferMemory  # type: ignore
from google.cloud import translate_v2 as translate
import os


DB_FAISS_PATH = "vectorstores/db_faiss"
conversation_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/jose/Downloads/massive-catfish-411714-89ecb4eed938.json"


custom_prompt_template = """Use the following information to answer the user's question.
You're designed to be a helpful chatbot that infers or tries to determine what medical condition the user may have
based on the symptoms gived. 

Please try to avoid using the phrase "I don't know" or "I don't know what to do".
Please try to be as specific and accurate as possible.

No matter if you are not a doctor, you can make a diagnosis based on the symptoms the user gives you and recommend some 
specialists the user can consult.

If the user asks you a question in another language that is not English, please follow these steps:
1. Detect the language the user is using.
2. Translate the question to English.
3. Answer the question in English.
4. Translate your English response to the language the user is using.

If you don't know the answer, please type "Currently I don't have enough information to answer your question.", don't
try to make up an answer.

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else
Helpful Answer:
"""


# --- TRANSLATION FUNCTIONS ---
def detect_language(text):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    return identifier.classify(text)[0]


def translate_google(text, target_language):
    client = translate.Client()
    result = client.translate(text, target_language=target_language)
    return result['translatedText']


# --- PROMPTS AND BOT FUNCTIONS ---
def set_custom_prompt():
    """ 
    Prompt template for QA retrieval for each vectorstore
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt


def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q4_K_S.bin",
        model_type="llama",
        config={
            'max_new_tokens': 2500,
            'temperature': 0.001,  # -> unpredicatbility | Higher = Less predictable "Nonsense"
            'context_length': 3500,
        }
    )
    return llm


def retrieval_qa_chain(llm, prompt, db):  # -> creates a question-answering chain (retrieval)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),  # -> retrieves the top 2 docs from db
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def qa_bot():  # -> creates a question-answering bot
    embeddings = HuggingFaceEmbeddings(model_name='distilbert-base-nli-mean-tokens', 
                                       model_kwargs= {'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()

    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa
