import os
from datasets import Dataset
from langchain_ollama.llms import OllamaLLM
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
# Set custom temporary directories
os.environ['TMPDIR'] = '/fred/oz345/Team3ProjectC/tmp'
os.environ['TEMP'] = '/fred/oz345/Team3ProjectC/tmp'
os.environ['TMP'] = '/fred/oz345/Team3ProjectC/tmp'

# Set Ollama data directory
# Initialize the language model
langchain_llm = OllamaLLM(
    model="mistral",
    temperature=0.1,
    num_ctx=8096,
)

# Initialize the embeddings
langchain_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"token": "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"}
)

def preprocess_dataframe(df):
    df['ground_truth'] = df['ground_truth'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else str(x)
    )
    df['question'] = df['question'].astype(str)
    df['answer'] = df['answer'].astype(str)
    df['contexts'] = df['contexts'].apply(
        lambda x: [str(item) for item in x] if isinstance(x, list) else [str(x)]
    )
    return df

# Read and preprocess the dataset
df = pd.read_csv('/fred/oz345/Team3ProjectC/data/halubench_covidQA_EnhancedRAG_Ollama_Colbert2.csv')
df = preprocess_dataframe(df)

# Prepare the data for evaluation
ragas_data = {
    "question": df['question'].tolist(),
    "contexts": df['contexts'].tolist(),
    "answer": df['answer'].tolist(),
    "ground_truth": df['ground_truth'].tolist()
}

ragas_dataset_llm = Dataset.from_dict(ragas_data)

# Evaluate the model with exception handling
try:
    results = evaluate(
        ragas_dataset_llm,
        metrics=[faithfulness, answer_relevancy, answer_correctness],
        llm=langchain_llm,
        embeddings=langchain_embeddings
    )
    result_llm = results.to_pandas()
    result_llm.to_csv('ragas_results_llm.csv', index=False)
    print(results)
except Exception as e:
    print(f"An error occurred during evaluation: {e}")
