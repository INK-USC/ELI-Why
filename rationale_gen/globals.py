# globals.py
# This file contains global variables and configurations used across rationale_gen folder.

DATASET_PATH = "../ELI_Why.jsonl"
    
# Configuration for each model from LM Studio or OpenAI API
MODEL_CONFIGS = {
    "Llama3.2": {
        "api_url": "http://127.0.0.1:1234/v1/chat/completions",
        "model": "llama-3.2-3b-instruct@4bit",
        "headers": {"Content-Type": "application/json"},
    },
    "Qwen2.5": {
        "api_url": "http://127.0.0.1:1234/v1/chat/completions",
        "model": "qwen2.5-14b-instruct",
        "headers": {"Content-Type": "application/json"},
    },
    "Gemma3": {
        "api_url": "http://127.0.0.1:1234/v1/chat/completions",
        "model": "lmstudio-community/gemma-3-27b-it",
        "headers": {"Content-Type": "application/json"},
    },
    "R1_Distilled_Llama": {
        "api_url": "http://127.0.0.1:1234/v1/chat/completions",
        "model": "deepseek-r1-distill-llama-8b",
        "headers": {"Content-Type": "application/json"},
    }
}

phd_prompt = """You will be asked a "Why" question. You are an expert in the domain of the why question you are asked. The user asking you the question also has a PhD in the domain of the question they asked.

Your job as an expert is to provide a concise explanation to the PhD holder. Make sure that the explanation is useful to the user - they will use it to validate and cross check important information. They may also use the explanation to teach that topic to a class.

Just provide the explanation as is - do not add any additional text like greetings or ornamental words.
"""

high_school_prompt = """You will be asked a "Why" question. You are an expert in the domain of the why question you are asked. The user asking you the question is someone who holds a basic american high school education. You can assume they are a "layperson" to the domain of the question asked.

Your job as an expert is to provide a concise explanation to the user. They asked you the question as they were curious about the topic, so make sure that the explanation is useful to them.

Just provide the explanation as is - do not add any additional text like greetings or ornamental words.
"""

elementary_school_prompt = """You will be asked a "Why" question. You are an expert in the domain of the why question you are asked. The user asking you the question is someone who holds a basic american elementary school education.

Your job as an expert is to provide a concise explanation to the user.

Just provide the explanation as is - do not add any additional text like greetings or ornamental words.
"""

default_prompt = """You will be asked a "Why" question. You are an expert in the domain of the why question you are asked.

Your job as an expert is to provide a concise explanation to the user.

Just provide the explanation as is - do not add any additional text like greetings or ornamental words.
"""


