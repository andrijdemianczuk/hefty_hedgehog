# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Edmonton City Council Memo Search
# MAGIC <img class="image" src="https://images.squarespace-cdn.com/content/v1/5ee7db99428b241c08b4f593/1595979740370-L602O16QVJJ32HSTABR0/Dub-Architects-Edmonton-City-Hall_03.jpg" width=500 />
# MAGIC
# MAGIC This dataset includes all memos sent to Edmonton City Council by Adminstration from January 1, 2022 through September 1, 2023.
# MAGIC
# MAGIC The full dataset can be found [here](https://data.edmonton.ca/City-Administration/Memos-to-Council/kyni-rzw4)
# MAGIC
# MAGIC The dataset includes a subset of 1,673 memos covering a variety of municipal affairs. The motivation for this project is to leverage newer indexing methodologies to improve document search with the use of language model pipelines to serve as a more efficient retrieval and summary workflow.
# MAGIC
# MAGIC The intention is to set up a precursor for further development of a large-language model that will be used to summarize responses in a human-like way with the expectation of conversation chaining in the future.
# MAGIC
# MAGIC
# MAGIC
# MAGIC <style>
# MAGIC .image{
# MAGIC   padding: 10px;
# MAGIC   float: left;
# MAGIC }
# MAGIC .right_box{
# MAGIC   margin: 30px; box-shadow: 10px -10px #CCC; width:650px; height:300px; background-color: #1b3139ff; box-shadow:  0 0 10px  rgba(0,0,0,0.6);
# MAGIC   border-radius:25px;font-size: 35px; float: left; padding: 20px; color: #f9f7f4; }
# MAGIC .badge {
# MAGIC   clear: left; float: left; height: 30px; width: 30px;  display: table-cell; vertical-align: middle; border-radius: 50%; background: #fcba33ff; text-align: center; color: white; margin-right: 10px; margin-left: -35px;}
# MAGIC .badge_b { 
# MAGIC   margin-left: 25px; min-height: 32px;}
# MAGIC </style>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Introduction
# MAGIC In this notebook, we're going to use the vector database we created with our embeddings in the previous notebook and use those embeddings to perform a document search based on conversational proximity. Rather than using term or word frequency, we will be relying on semantic (meaning) relationships that established by language modelling.
# MAGIC
# MAGIC The process is as straightforward as asking the vector db for 'n-number of closest documents'. This is evaluated against a summary of the english query passed in by the user.

# COMMAND ----------

# MAGIC %pip install -U chromadb==0.3.22 langchain==0.0.164 transformers==4.29.0 accelerate==0.19.0 bitsandbytes==0.41.1 einops==0.6.1 xformers==0.0.20

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=ademianczuk $db=council_memos

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating an Instance of Our Vector Database in Memory
# MAGIC <img src="https://techcommunity.microsoft.com/t5/image/serverpage/image-id/469397i17D9AC793531E359/image-size/large?v=v2&px=999" width=750/><br/>
# MAGIC First, we'll need to load all of our parameters based on where we persisted our vector database in the previous notebook. We will also be loading the model we used to translate the embeddings to decode them ([all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)).
# MAGIC
# MAGIC This sets up all of the required paramters for document retrieval

# COMMAND ----------

# DBTITLE 1,Load the Vector DB From the Last Notebook
# Start here to load a previously-saved DB
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

if len(get_available_gpus()) == 0:
    Exception(
        "Running inference without GPU will be slow. We recommend you switch to a Single Node cluster with at least 1 GPU to properly run this demo."
    )

memo_vector_db_path = "/dbfs" + model_path + "/vector_db"

hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(
    collection_name="memo_docs",
    embedding_function=hf_embed,
    persist_directory=memo_vector_db_path,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Invoking a Search Against Our Vector Database
# MAGIC Invoking the search for texts within our vector database is as simple as saying 'I want n-number of documents that are most closely related to this term'. We'll get a list of results that we can then simply present to the user or return from the calling application.

# COMMAND ----------

def get_similar_docs(question, similar_doc_count):
    return db.similarity_search(question, k=similar_doc_count)


# Let's test it
i = 0
for doc in get_similar_docs("What is the latest LRT development?", 2):
    i += 1
    print("\r\n")
    print(f"--- document # {i} ----")
    print(doc.page_content)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Document similarity search
# MAGIC <img class="image" src="https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/SemanticSearch.png"/>
# MAGIC So this is great! We received the two most likely documents to contain information relevant to our intial query. How did this happen? When we invoked the instance of our vector db we also passed in the encoding model that we used when we defined the parameters for our db in the first place. This is important not only for decoding, but to ensure that the same model was used to infer the translation of the query text. This means that the same pipeline that created the embeddings in our database was also used to encode our query. Then, through the process of evaluating semantics, our vector database returned the two closest embeddings!
# MAGIC
# MAGIC So now we have some things to consider:
# MAGIC
# MAGIC - We can end here, or we can keep building
# MAGIC - Doc search is cheap and easy to use, without a full-blown LLM
# MAGIC - Vector searches are generally very fast and don't require a lot of resources compared to full large-language machines
# MAGIC - Next step would be to summarize the results and translate them (why not?)
# MAGIC - We can use LangChain to string together queries based on result summaries
# MAGIC
# MAGIC <style>
# MAGIC .image{
# MAGIC   padding: 20px;
# MAGIC   float: right;
# MAGIC }
# MAGIC </style>
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt engineering with LangChain
# MAGIC LangChain is a framework for developing applications powered by language models. LangChain is widely accepted as the means to connect a language model to other sources of data and allow a language model to interact with its environment.
# MAGIC
# MAGIC There are two main value props the LangChain framework provides:
# MAGIC
# MAGIC 1. Components: LangChain provides modular abstractions for the components neccessary to work with language models. LangChain also has collections of implementations for all these abstractions. The components are designed to be easy to use, regardless of whether you are using the rest of the LangChain framework or not.
# MAGIC
# MAGIC 1. Use-Case Specific Chains: Chains can be thought of as assembling these components in particular ways in order to best accomplish a particular use case. These are intended to be a higher level interface through which people can easily get started with a specific use case. These chains are also designed to be customizable.
# MAGIC
# MAGIC
# MAGIC For a brief description on how to get started with LangChain and understand the parameters used for the pipeline that we're assembling for the model, have a quick look at the LangChain quickstart:
# MAGIC
# MAGIC <a href="https://python.langchain.com/docs/get_started/quickstart" target="_blank" rel="noopener noreferrer">LangChain Quickstart</a>
# MAGIC
# MAGIC For our use case, we will be relying on the chain loader class for the Q&A to format the results in an answer format and the summarize class that truncates the results in a human-readable way. LangChain can also provide us with a framework that we can use for context embedding to help keep track of the conversation with the machine.

# COMMAND ----------

# DBTITLE 1,Building the QA Chain and Issuing a Prompt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain


def build_qa_chain():
    torch.cuda.empty_cache()

    # Decide on the model we want to use
    model_name = "databricks/dolly-v2-7b"  # can use dolly-v2-3b or dolly-v2-7b for smaller model and faster inferences.

    # Define the pipeline to be used with our model (this will change depending on the model of our choosing)
    instruct_pipeline = pipeline(
        model=model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        return_full_text=True,
        max_new_tokens=256,
        top_p=0.95,
        top_k=50,
    )

    # Create the context for the discussion
    template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  Instruction: 
  You are an analyst and your job is to help find the right council memos. 
  Use only information in the following paragraphs to answer the question at the end. Explain the answer with reference to these paragraphs. If you don't know, say that you do not know.

  {context}
 
  Question: {question}

  Response:
  """

    # Create the message prompt
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    # Create an instance of the hugging face pipeline
    hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)

    # Pass all everything into LangChain and return the QA object to the caller
    return load_qa_chain(llm=hf_pipe, chain_type="stuff", prompt=prompt, verbose=True)

# COMMAND ----------

# DBTITLE 1,Create an Instance of the QA Chain Object
# Building the chain will load the model for tuning and can take several minutes depending on the model size
qa_chain = build_qa_chain()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Translating and Summarizing the Results
# MAGIC
# MAGIC Now that we have both a document similarity search and a translation engine availble, let's combine the two for a summarized response in an english-like way. This is a two-step process:
# MAGIC
# MAGIC 1. First get the top n-number of similar docs related to the question we asked the machine
# MAGIC
# MAGIC 1. Then, send the top documents into the QA chain to run through the summary pipeline. This is the qa_chain() function that assembles the documents and returns a result based on the pipeline we built for our model, the type of document chain and the using the template for the prompt we can guarantee that the inputs will always be consistent.

# COMMAND ----------

# DBTITLE 1,Using the Similar Docs to Answer the Question
def answer_question(question):
    similar_docs = get_similar_docs(question, similar_doc_count=2)
    # print(similar_docs)
    result = qa_chain({"input_documents": similar_docs, "question": question})
    result_html = f'<p><blockquote style="font-size:24">{question}</blockquote></p>'
    result_html += f"<p><blockquote style=\"font-size:18px\">{result['output_text']}</blockquote></p>"
    result_html += "<p><hr/></p>"
    for d in result["input_documents"]:
        source_id = d.metadata["source"]
        result_html += f"<p><blockquote>{d.page_content}<br/></blockquote></p>"
    displayHTML(result_html)

# COMMAND ----------

answer_question("Is the LRT safe to ride?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC Great! So now we have a full system that not only allows us to ask a question in a human-like way, but also get a human-like response based on a set of documents returned from out vector database!
# MAGIC
# MAGIC So in summary, we've now managed to accomplish:
# MAGIC
# MAGIC - Encoding our documents as vector embeddings
# MAGIC
# MAGIC - Storing our embeddings in an efficient vector database
# MAGIC
# MAGIC - Translated a user query into a dense vector for search
# MAGIC
# MAGIC - Query and retrieve the most relevant documents based on the user's query
# MAGIC
# MAGIC - Summarized and translated the returned documents into an english-like summarized response
# MAGIC
# MAGIC ## Where Do We Go Next?
# MAGIC So now that we have a working transformer-based machine (with a full encoder, process and decoder stack), we can start extending this to new use cases and implementations. One of the most relevant design concepts of modern software engineering states that objects must be decoupled, encapsulated, polymorphic and inheritable. In other words, the components must be modular. With that in mind, there are a couple of key features that will be added to this project in the future:
# MAGIC
# MAGIC - The ability to hot-swap models based on pipeline templates
# MAGIC
# MAGIC - Changing the document inference type to summary chain on a conditional basis of the query semantics
# MAGIC
# MAGIC - Converting all functions to a class-based API
# MAGIC
# MAGIC - Support for experimentation and benchmarking for desired targets
# MAGIC
# MAGIC So, let's keep building!

# COMMAND ----------

# Make sure you restart the python kernel to free our gpu memory if you're using multiple notebooks0
# (load the model only once in 1 single notebook to avoid OOM)
dbutils.library.restartPython()
