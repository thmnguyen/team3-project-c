import os

# Set environment variables before importing transformers
os.environ['HF_HOME'] = '/fred/oz345/Team3ProjectC/.cache/huggingface_home'
os.environ['HF_HUB_CACHE'] = '/fred/oz345/Team3ProjectC/.cache/huggingface_hub'

import pandas as pd
from tqdm import tqdm
# Updated imports from langchain
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Ensure GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the local model directory
model_dir = "/fred/oz345/Team3ProjectC/models/mistralai"

print('Loading tokenizer...')
# Load the tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    local_files_only=True,
    cache_dir='/fred/oz345/Team3ProjectC/.cache/transformers'
)

print('Loading model...')
# Load the model from the local directory
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    local_files_only=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    cache_dir='/fred/oz345/Team3ProjectC/.cache/transformers'
).to(device)

print('Model loaded.')

# Create the Hugging Face pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,  # Set the device
    max_new_tokens=512,
    do_sample=True,
    temperature=0.1,  # Set temperature to a positive value
)

# Wrap the pipeline into a LangChain LLM
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Load Hugging Face Embeddings model with device specified
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={
        "device": device,  # Ensure embeddings model uses GPU
    }
)

# Load and process data
df = pd.read_csv('/fred/oz345/Team3ProjectC/halubench1.csv', sep='\t')
df = df.rename(columns={'answer': 'ground_truth', 'passage': 'contexts'})
# Process the entire dataset or a subset for testing
# df = df[:3]  # Uncomment for testing with first 3 rows

# Function to apply the RAG pipeline to a batch of rows
def batch_calculation(batch_df, llm=llm, embed_model=embed_model):
    # Combine all contexts
    all_contexts = batch_df['contexts'].tolist()

    # Split the contexts into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=16)
    all_splits = []
    for context in all_contexts:
        splits = text_splitter.split_text(context)
        all_splits.extend(splits)

    # Convert each text chunk into a Document object
    documents = [Document(page_content=text) for text in all_splits]

    # Create a vectorstore from the documents
    vectorstore = FAISS.from_documents(documents, embed_model)

    # Define the prompt template
    prompt_template = """
You are a helpful AI assistant. Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know; don't try to make up an answer.
Context: {context}
Question: {question}
Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create the RAG pipeline
    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    # Prepare inputs for batch processing
    questions = batch_df['question'].tolist()
    results = []

    for question in tqdm(questions, desc="Processing questions"):
        # Use the updated method `invoke` instead of `__call__`
        result = rag_pipeline.invoke({"query": question})
        results.append(result['result'])

    return results

# Apply the calculation function to batches
batch_size = 1500  # Adjust based on your GPU memory
df['answer'] = ''

for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
    batch_df = df.iloc[i:i+batch_size]
    batch_results = batch_calculation(batch_df)
    df.loc[batch_df.index, 'answer'] = batch_results
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    print(f"Batch {i // batch_size + 1}: Allocated={allocated / (1024**3):.2f} GB, Reserved={reserved / (1024**3):.2f} GB")

# Export the result to a CSV file
df.to_csv('result_rag_halubench.csv', index=False)

print("Processing complete! Results saved to result_rag_halubench.csv")
