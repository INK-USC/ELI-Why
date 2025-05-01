import argparse
import pandas as pd
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
import nltk
import torch

# Download required NLTK data
nltk.download('wordnet')
nltk.download('punkt')

# Assume these prompts are imported from globals.py
from globals import phd_prompt, high_school_prompt, elementary_school_prompt, default_prompt

# Load the JSONL file into a DataFrame
def load_jsonl_to_dataframe(file_path: str) -> pd.DataFrame:
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Define the list of prompts and corresponding column names.
prompts = [phd_prompt, high_school_prompt, elementary_school_prompt, default_prompt]
prompt_names = ["Graduate School", "High School", "Elementary School", "Default"]

# Map available model names to their VLLM identifiers.
VLLM_MODELS = {
    "Qwen2.5": "qwen2.5-14b-instruct",
    "Gemma3": "lmstudio-community/gemma-3-27b-it",
    "Llama3.2": "meta-llama/llama-3.2-3B-Instruct"
}

def generate_explanations_for_questions(
    df: pd.DataFrame,
    question_col: str,
    prompts: list,
    prompt_names: list,
    llm: LLM,
    sampling_params: SamplingParams,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    For each question in df[question_col], generate explanations using the given prompts.
    Uses batch calls to VLLM for efficiency.
    """
    # Prepare new columns for each prompt.
    for p_name in prompt_names:
        df[p_name] = ""

    # For each prompt, build a list of conversation histories and get responses in batch.
    for prompt, p_name in zip(prompts, prompt_names):
        retries = 0
        while retries < max_retries:
            try:
                conversation_list = []
                for question in df[question_col]:
                    conversation = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": question}
                    ]
                    conversation_list.append(conversation)

                # Batch call to VLLM.
                responses = llm.chat(conversation_list, sampling_params)
                for i, resp in enumerate(responses):
                    df.at[i, p_name] = resp.outputs[0].text.strip()
                break  # Successful batch call.
            except Exception as e:
                print(f"Batch request error for prompt '{p_name}', retry {retries+1}: {e}")
                retries += 1
        if retries == max_retries:
            df[p_name] = "Error after max_retries"
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate rationales using VLLM with 4-bit quantization.")
    parser.add_argument("--model", choices=list(VLLM_MODELS.keys()), required=True,
                        help="Select the model to use.")
    args = parser.parse_args()

    # Set input and output paths.
    input_path = "../ELI_Why.jsonl"
    save_path = f"{args.model}_rationales.csv"

    # Load the JSONL file.
    df = load_jsonl_to_dataframe(input_path)
    print("Loaded questions:")
    print(df.head())

    # Initialize the VLLM model with 4-bit quantization.
    # (Assuming the LLM constructor accepts 'quantize' and 'quantize_bits' parameters.)
    model_identifier = VLLM_MODELS[args.model]
    llm = LLM(model=model_identifier, max_model_len=8192, dtype=torch.bfloat16, quantization="bitsandbytes")
    sampling_params = SamplingParams(temperature=0.1, max_tokens=4096)

    print("Generating rationales using model:", args.model)
    df = generate_explanations_for_questions(
        df=df,
        question_col="Question",
        prompts=prompts,
        prompt_names=prompt_names,
        llm=llm,
        sampling_params=sampling_params
    )

    # Save the results.
    df.to_csv(save_path, index=False)
    print("Rationales saved to", save_path)

if __name__ == "__main__":
    main()
