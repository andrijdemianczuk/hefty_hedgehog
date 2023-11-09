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
# MAGIC
# MAGIC In this notebook we will be taking a collection of memos from Administration sent to Edmonton City Council and creating a summary, an encoding and converting the encodings to embeddings to be persisted for later recall by our language machine. The process of engineering an pre-processing the data is critical to ensuring that the corpus of data doesn't confuse the tuning process in the pipeline defintion stage later on.
# MAGIC
# MAGIC In order to facilitate this, we will require a few external libraries to help us with this work.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install -U chromadb==0.3.22 langchain==0.0.164 transformers==4.29.0 accelerate==0.19.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Libraries Used
# MAGIC We will be using a small set of libraries to help is with our encoding and embedding tasks:
# MAGIC
# MAGIC <img src="https://blog.langchain.dev/content/images/2023/02/langchain-chroma-light.png" width=500>
# MAGIC <img src="https://www.thesoftwarereport.com/wp-content/uploads/2023/09/Hugging-Face2.png" width=500>
# MAGIC
# MAGIC ### ChromaDB
# MAGIC ChromaDB will be the vector database that we'll be using for our document embedding storage. ChromaDB is a highly-efficient in-memory database that supports a high-degree of concurrency and parallelism, allowing us to take full advantage of our Spark Cluster.
# MAGIC
# MAGIC ### Transformers
# MAGIC Transformers is a library developed by Hugging Face which will grant us a few benefits; transformers is an easy way to source and download models from the Hugging Face model repository and will help provide us with the dataset pipeline framework with options that can be tuned for the desired outcome.
# MAGIC
# MAGIC ### LangChain
# MAGIC LangChain will provide us with much of the core functionality used when creating our document summaries and structure our vector arrays that we will be using for storage in our Vector Database. LangChain provides us the the framework that allows a Q&A-style dialog with the machine. LangChain provides a number of interfaces that can naturally be extended based on business need

# COMMAND ----------

# DBTITLE 1,Imports
#Pyspark common functions
from pyspark.sql.functions import *

#Supporting ML Libraries
from typing import Iterator
from transformers import pipeline

#Pandas utilities
import pandas as pd

#HFEmbeddings to interface with HF-hosted models
from langchain.embeddings import HuggingFaceEmbeddings

#For vectorizing and storing docs in Chroma
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

# COMMAND ----------

# DBTITLE 1,Initialize Supporting Functions
# MAGIC %run ./_resources/00-init $catalog=ademianczuk $db=council_memos

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Data Pre-Processing
# MAGIC
# MAGIC <img class="image" src="https://images.prismic.io/coresignal-website/c3225f2b-9991-4ebe-a5a6-26542a929052_Data+Transformation+Benefits_+Types%2C+and+Processes.jpg?auto=compress%2Cformat&fit=max&q=75">
# MAGIC
# MAGIC Data pre-processing is a critical task both in data engineering and data science when getting ready for ML workloads. We need to ensure that our data is consistent, error-free and in a format that will be ready for experimentation and development.
# MAGIC
# MAGIC Generally speaking, we're going to look for a number of factors but not limited to:
# MAGIC * Elimination of null values
# MAGIC * Removing stop words, symbols etc.
# MAGIC * Language transformation and formatting
# MAGIC * Formatting for illegal characters
# MAGIC
# MAGIC There is no hard-and-fast rule around pre-processing. The goal is to just make it as consistent and as reliable as possible.
# MAGIC
# MAGIC <style>
# MAGIC .image{
# MAGIC   padding: 10px;
# MAGIC   float: right;
# MAGIC   width: 500px;
# MAGIC }

# COMMAND ----------

# DBTITLE 1,Load The Data From Source
df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(f"/Volumes/{catalog}/{db}/memos_src/Memos_to_Council.csv")
)

df = (
    df.withColumnRenamed("Date of Memo", "Date_of_Memo")
    .withColumnRenamed("TO (one per line)", "TO")
    .withColumnRenamed("FROM (one per line)", "FROM")
    .withColumnRenamed("Memo Subject", "Memo_Subject")
    .withColumnRenamed("FOIP Exception(s) (one per line)", "FOIP_Exceptions")
    .withColumnRenamed("Link to Memo (file upload)", "Link_to_Memo")
    .withColumnRenamed("Memo Text Part 1", "Memo_Text_1")
    .withColumnRenamed("Memo Text Part 2", "Memo_Text_2")
    .withColumnRenamed("Is this Memo Related to a Meeting?", "Meeting_Related")
    .withColumnRenamed("Meeting Date", "Meeting_Date")
    .withColumnRenamed("Meeting Item Number(s) (one per line)", "Meeting_Item_Numbers")
    .withColumnRenamed(
        "Committee or Council Name(s) (one per line)", "Committee_or_Council_Names"
    )
    .withColumn("Memo_Text_1", regexp_replace("Memo_Text_1", r"\\s+|●|[A-Z]\/g", " "))
    .withColumn("Memo_Text_2", regexp_replace("Memo_Text_2", r"\\s+|●|[A-Z]\/g", " "))
    .withColumn("Memo_Text_1", regexp_replace("Memo_Text_1", r"\\s+|■|[A-Z]\/g", " "))
    .withColumn("Memo_Text_2", regexp_replace("Memo_Text_2", r"\\s+|■|[A-Z]\/g", " "))
    .withColumn("Memo_Text_1", regexp_replace("Memo_Text_1", r"\\s+|➔|[A-Z]\/g", " "))
    .withColumn("Memo_Text_2", regexp_replace("Memo_Text_2", r"\\s+|➔|[A-Z]\/g", " "))
)

df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
    f"{catalog}.{db}.memos_raw"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dimensionality Reduction & String Conversion
# MAGIC <img src="https://static.wixstatic.com/media/a8b2bc_71f2c4756c0447d2978e1ea1316a7a6f~mv2.png/v1/fit/w_997%2Ch_771%2Cal_c%2Cq_80,enc_auto/file.jpg" width=300/><br/><br/>
# MAGIC In the second stage of pre-processing, we are taking the data as it was captured (raw) and cleaning it up a bit. We're going to focus on two key areas here for our dataset:<br/><br/>
# MAGIC 1. **Dimensionality Reduction**<br/>
# MAGIC By reducing the number of dimensions in our table, we focus only on the key elements required of the dataset. Not only does this shrink down the size of the table on-disk, but it focuses solely on the task-at-hand and everything that's required of the table when working in-memory. For every dimension that's present in a collection, the cost and difficulty of processing the data increases exponentially. Although more verbose, this comes at a cost. This is otherwise known as *the curse of dimensionality*. This concept applies both to Data Engineering and Data Science.
# MAGIC 1. **String Conversion**<br/>
# MAGIC Whenever possible, we try to reduce our StringTypes to a primitive or near-primitive type. This is for a number of reasons, including typing expectations, constraints, lowered cardinality and transformational efficiency. Strings are by virtue fairly unstructured and highly cardinal which adds to the complexity of processing (which is why we're even considering vectorization of our documents in the first place ;))
# MAGIC
# MAGIC <style>
# MAGIC .image{
# MAGIC   padding: 10px;
# MAGIC   float: right;
# MAGIC   width: 500px;
# MAGIC }

# COMMAND ----------

# DBTITLE 1,Clean & Store The Silver Table
# Prepare the training dataset by concatentating the memos into a single column and converting the date columns to timestamps for easy processing.

df = spark.table(f"{catalog}.{db}.memos_raw")

df = (
    df.select(
        col("Timestamp"),
        col("Date_of_Memo"),
        col("Link_to_Memo"),
        col("Memo_Text_1"),
        col("TO"),
        col("FROM"),
    )
    .toDF("Timestamp", "Date_of_Memo", "Link_to_Memo", "Text", "TO", "FROM")
    .withColumn("Timestamp", to_timestamp("Timestamp", "MMM dd, yyyy hh:mm a"))
    .withColumn("Date_of_Memo", to_timestamp("Date_of_Memo", "MMM dd, yyyy"))
)

df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
    f"{catalog}.{db}.memos_silver"
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## What is Vectorization and Why is it So Important?
# MAGIC
# MAGIC <img class="image" src="https://dornerworks.com/wp-content/uploads/2019/12/vectorized-instruction.png">
# MAGIC
# MAGIC Vectorization is basically the process of converting data elements to a numerical representation. This can happen at the data or process-level and can lead to significant performance improvements in many algorithms, particularly in mathematical and computational tasks. We can think of pandas dataframes on pyspark as data elements as well.
# MAGIC
# MAGIC Vectorization is actually going to be used in two different contexts in this project. The most obvious is with regards to indexing the embeddings, but an important second context is around high-performance processing. For the case of our udfs, we will be leveraging something called a pandas_udf which allows us to vectorize a process on a data structure. This allows us to leverage a high-degree of parallelism due to the fact that pandas on pyspark is very CPU and GPU efficient. Processes can be spread across the entire array compute slots for parallel processing when compared to pyspark.sql udfs.
# MAGIC
# MAGIC Programming Vectorized functions is generally done through an abstraction layer (an API) that translates the request into some type of material execution. This process is commonly referred to as *declarative programming*. In other words, we're generally less concerned about the 'how' and more concerned about the 'what'. 
# MAGIC
# MAGIC <style>
# MAGIC .image{
# MAGIC   padding: 10px;
# MAGIC   float: right;
# MAGIC   width: 500px;
# MAGIC }

# COMMAND ----------

# DBTITLE 1,Create the Training Dataset
# This function is only programmed for a single GPU, single instance node. We can look into fixing this later for multi-gpu support (collect)
@pandas_udf("string")
def summarize(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # Load the model for summarization
    torch.cuda.empty_cache()
    summarizer = pipeline(
        "summarization", model="sshleifer/distilbart-cnn-12-6", device_map="auto"
    )

    def summarize_txt(text):
        if len(text) > 10000:
            return summarizer(text)[0]["summary_text"]
        return text

    for series in iterator:
        yield series.apply(summarize_txt)


df2 = spark.table(f"{catalog}.{db}.memos_silver")
#df2 = df2.repartition(1).withColumn("text_short", summarize("text")) #This is not multi-gpu optimized

df2.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
    f"{catalog}.{db}.memos_training_dataset"
)
display(spark.table(f"{catalog}.{db}.memos_training_dataset"))

# COMMAND ----------

# DBTITLE 1,Download the Sentence Embedding Model Used For Encoding
hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# COMMAND ----------

# DBTITLE 1,Initialize the Directory for ChromaDB
# Prepare a directory to store the document database. Any path on `/dbfs` will do.
dbutils.widgets.dropdown(
    "reset_vector_database",
    "false",
    ["false", "true"],
    "Recompute embeddings for chromadb",
)
memo_vector_db_path = model_path + "/vector_db"

# Don't recompute the embeddings if the're already available
compute_embeddings = dbutils.widgets.get(
    "reset_vector_database"
) == "true" or is_folder_empty(memo_vector_db_path)

if compute_embeddings:
    print(f"creating folder {memo_vector_db_path} under our blob storage (dbfs)")
    dbutils.fs.rm(memo_vector_db_path, True)
    dbutils.fs.mkdirs(memo_vector_db_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## A Bit About Text Embedding
# MAGIC
# MAGIC ### Sparse Vectors
# MAGIC <img class="image" src="https://images.prismic.io/deepset/25e1c1fe-39de-4965-9fc3-b65159a90983_tobeornot.png?auto=compress,format" width=500 />
# MAGIC <img class="image" src="https://images.prismic.io/deepset/d8a05d0c-8ffa-4f95-9f4a-9b258bf2cfa3_bagofowrds.png?auto=compress,format" width=500 />
# MAGIC
# MAGIC By embedding a word or a longer text passage as a vector, it becomes manageable by computers, which can then, for example, compute how similar two pieces of text are to each other. The key here of course, is a numerical representation of an object. The term 'vector' implies a degree of dimensionality that allows one object to be quickly compared to others. When paired with other classic methodologies (such as clustering, frequency classification etc.) we can start to understand the structure and relationships of embeddings.
# MAGIC
# MAGIC A typical 4-bit sparse vector will look like:
# MAGIC ```
# MAGIC [[2,1,1,2],[2,0,1,1],[3,0,2,1],[0,0,0,1].....]
# MAGIC ```
# MAGIC
# MAGIC The challenge with sparse vectors, is that all words are treated equally without regard for semantics or positional encoding.
# MAGIC
# MAGIC ### But What About Dense Vectors?
# MAGIC
# MAGIC <img src="https://images.prismic.io/deepset/07db31a8-0490-4611-be31-033c8832c629_grouping.png?auto=compress,format" width=500 />
# MAGIC <img src="https://images.prismic.io/deepset/f7044611-93e0-49e5-8b72-765614d9a56d_vectorspace_sentences.png?auto=compress,format" width=500 />
# MAGIC
# MAGIC Dense vectors address the issue of weighting and bias by containing mostly non-zero elements in their embeddings. Dense vectors have several advantages over sparse vectors, but three of the most notable ones include:
# MAGIC 1. Ascribing meaning (e.g., 'he' can refer to a male individual)
# MAGIC 2. Previously unseen words can be represented as part of the embedding matrix
# MAGIC 3. Semantics can be encoded and trained
# MAGIC
# MAGIC A typical dense vector will look like:
# MAGIC ```
# MAGIC [0.01,0.85,0.17,0.38....]
# MAGIC ```
# MAGIC
# MAGIC However, this all comes at a cost; dense vectors are much more expensive to convert and infer than sparse vectors. This is a big reason why the hyper-parallelism of gpu-enabled processing with low-level frameworks CUDA (implemented via PyTorch for example) have had such a big impact on the AI & ML industries.
# MAGIC
# MAGIC That’s why semantic search is often used to manage large collections of documents through an intuitive search function. It is also the basis for advanced question answering (QA) systems. Imagine, for example, that you’re looking for an answer from a corpus of emails. You know that the answer to your question is somewhere in there — you just don’t know where exactly. Since we are working with a reasonable collection of memos to city council as the corpus of data, encoding these texts as dense vectors makes them easy to search and summarize :)
# MAGIC
# MAGIC <style>
# MAGIC .image{
# MAGIC   padding: 10px;
# MAGIC   float: left;
# MAGIC   width: 500px;
# MAGIC }

# COMMAND ----------

# DBTITLE 1,Create the Object Embeddings and Persist the DB For Recall
all_texts = spark.table(f"{catalog}.{db}.memos_training_dataset")

print(f"Saving document embeddings under /dbfs{memo_vector_db_path}")

if compute_embeddings:
    # Remove rows with Null values in the 'text' field
    all_texts = all_texts.dropna(subset=["text"])

    # Transform our rows as langchain Documents
    documents = [
        Document(page_content=r["Text"], metadata={"source": r["Link_to_Memo"]})
        for r in all_texts.collect()
    ]

    # Init the chroma db with the sentence-transformers/all-mpnet-base-v2 model loaded from hugging face  (hf_embed)
    db = Chroma.from_documents(
        collection_name="memo_docs",
        documents=documents,
        embedding=hf_embed,
        persist_directory="/dbfs" + memo_vector_db_path,
    )
    
    db.similarity_search("dummy")  # tickle it to persist metadata (?)
    db.persist()

# COMMAND ----------

# Make sure you restart the python kernel to free our gpu memory if you're using multiple notebooks0
# (load the model only once in 1 single notebook to avoid OOM)
dbutils.library.restartPython()
