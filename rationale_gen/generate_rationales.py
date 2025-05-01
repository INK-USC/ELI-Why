import argparse
import pandas as pd
import json
from tqdm import tqdm
from lm_loader import create_model_instance
from globals import DATASET_PATH, phd_prompt, high_school_prompt, elementary_school_prompt, default_prompt

def load_jsonl_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a JSONL file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the JSONL file.
        
    Returns:
        pd.DataFrame: DataFrame containing the data from the JSONL file.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="Generate responses for 'Why' questions using a language model.")
    parser.add_argument("--model", choices=["Llama3.2", "Qwen2.5", "Gemma3", "R1_Distilled_Llama"], required=True,
                        help="Model to use for generating responses")
    args = parser.parse_args()

    # Load the JSONL file into a DataFrame
    df = load_jsonl_to_dataframe(DATASET_PATH)
    print("Loaded questions:")
    print(df.head())

    # Create a model instance based on the command-line argument
    model = create_model_instance(args.model)

    # Define prompt names and corresponding prompts
    prompt_names = ["Graduate School", "High School", "Elementary School", "Default"]
    prompts = [phd_prompt, high_school_prompt, elementary_school_prompt, default_prompt]

    all_answers = []
    # Iterate over each question in the DataFrame
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        question = row["Question"]
        answers_row = {"Question": question}
        # For each prompt, build a conversation and get the answer from the model
        for prompt_name, prompt in zip(prompt_names, prompts):
            conversation = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ]
            # Send the conversation to the model
            result = model.chat_completion(conversation, temperature=0.1, max_tokens=512)
            if result is not None and "choices" in result and result["choices"]:
                answer_text = result["choices"][0]["message"]["content"].strip()
            else:
                answer_text = ""
            # Column header combines the model name and prompt type
            col_name = f"{args.model} {prompt_name}"
            answers_row[col_name] = answer_text
        all_answers.append(answers_row)

    # Create a DataFrame with the answers and save it as a CSV file
    answers_df = pd.DataFrame(all_answers)
    save_path = f"{args.model}_rationales.csv"
    answers_df.to_csv(save_path, index=False)
    print(f"Responses saved to {save_path}")

if __name__ == "__main__":
    main()
