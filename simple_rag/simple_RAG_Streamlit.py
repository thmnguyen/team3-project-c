# This code implements simple RAG pipe line with LLM through Langchain LLama-Cpp. To run: Streamlit run Simple_RAG_Streamlit.py
import streamlit as st
import logging
import os
import tempfile
import shutil
from typing import List, Any, Optional
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager


# Streamlit page configuration
st.set_page_config(
    page_title="Local LLM Multi-PDF/TXT RAG Streamlit UI",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Local LLM configuration
LOCAL_MODEL_PATH = r"C:\Users\thmng\Documents\LLM"  # Update this path
LOCAL_MODELS = {
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf": r"C:\Users\thmng\Documents\LLM\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf": r"C:\Users\thmng\Documents\LLM\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    "Mistral-7B-v0.3.Q4_K_M.gguf": r"C:\Users\thmng\Documents\LLM\Mistral-7B-v0.3.Q4_K_M.gguf"
}

def create_vector_db(file_uploads: List[Any]) -> Chroma:
    logger.info(f"Creating vector DB from {len(file_uploads)} file uploads")
    temp_dir = tempfile.mkdtemp()
    all_docs = []

    for file_upload in file_uploads:
        path = os.path.join(temp_dir, file_upload.name)
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())
            logger.info(f"File saved to temporary path: {path}")
            
        if file_upload.name.lower().endswith('.pdf'):
            loader = UnstructuredPDFLoader(path)
        elif file_upload.name.lower().endswith('.txt'):
            loader = TextLoader(path)
        else:
            logger.warning(f"Unsupported file type: {file_upload.name}")
            continue
        
        try:
            data = loader.load()
            logger.info(f"Loaded {len(data)} documents from {file_upload.name}")
        except Exception as e:
            logger.error(f"Error loading {file_upload.name}: {str(e)}")
            continue

        if not data:
            logger.warning(f"No content extracted from {file_upload.name}")
            continue

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        
        if not chunks:
            logger.warning(f"No chunks generated from {file_upload.name}")
            continue

        logger.info(f"Generated {len(chunks)} chunks from {file_upload.name}")

        # Add metadata to each chunk
        for chunk in chunks:
            chunk.metadata["source"] = file_upload.name
        
        all_docs.extend(chunks)
        logger.info(f"Document {file_upload.name} split into {len(chunks)} chunks")

    if not all_docs:
        error_msg = "No valid documents to create vector database. Check the logs for details on each file processing attempt."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Total valid chunks across all documents: {len(all_docs)}")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_db = Chroma.from_documents(
        documents=all_docs, embedding=embeddings, collection_name="myRAG"
    )
    logger.info(f"Vector DB created with {len(all_docs)} total chunks")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db

def process_question(question: str, vector_db: Chroma, model_path: str) -> str:
    logger.info(f"Processing question: {question} using model: {model_path}")
    
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Initiate LLM
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.1,
        max_tokens=2048,
        n_gpu_layers=-1,
        n_ctx=4000,
        callback_manager=callback_manager,
        verbose=True,
    )

    # Vector store retriver with similarity search as default
    retriever = vector_db.as_retriever()

    # Template format for Llama 3.1 model. Other model should use different template format (https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template?fbclid=IwY2xjawFeHn9leHRuA2FlbQIxMAABHaUp1_XPVOfZj8dz6BRDPQILZsNu4LOIdYihwtQXwp7fq5jnjo6LAiH_yg_aem_yNUUwbESAZ3EAWxTgI1vmQ#supported-templates)
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Answer the question based ONLY on the following context:
    {context}<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Only provide the answer from the context, nothing else.
    Add snippets of the context you used to answer the question, and mention which file(s) the information came from.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response

def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        vector_db.delete_collection()
        st.session_state.pop("file_uploads", None)
        st.session_state.pop("vector_db", None)
        st.success("Collection and temporary files deleted successfully.")
        logger.info("Vector DB and related session state cleared")
        st.rerun()
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")

def main() -> None:
    """
    Main function to run the Streamlit application.

    This function sets up the user interface, handles file uploads,
    processes user queries, and displays results using the local LLM.
    """
    st.subheader("🦙 Local LLM Multi-PDF/TXT RAG playground", divider="gray", anchor=False)

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    selected_model = col2.selectbox(
        "Pick a local LLM model ↓", list(LOCAL_MODELS.keys())
    )

    file_uploads = col1.file_uploader(
        "Upload PDF or TXT files ↓", type=["pdf", "txt"], accept_multiple_files=True
    )

    if file_uploads:
        st.session_state["file_uploads"] = file_uploads
        if st.session_state["vector_db"] is None:
            try:
                st.session_state["vector_db"] = create_vector_db(file_uploads)
                st.success(f"{len(file_uploads)} file(s) uploaded and processed successfully.")
            except ValueError as e:
                st.error(f"Error processing files: {str(e)}")
                logger.error(f"Error in create_vector_db: {str(e)}")
                st.session_state["vector_db"] = None

        st.success(f"{len(file_uploads)} file(s) uploaded and processed.")

    delete_collection = col1.button("⚠️ Delete collection", type="secondary")

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    with col2:
        message_container = st.container(height=500, border=True)

        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                message_container.chat_message("user", avatar="😎").markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], LOCAL_MODELS[selected_model]
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload PDF or TXT files first.")

                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload PDF or TXT files to begin chat...")

if __name__ == "__main__":
    main()