import numpy as np
import pandas as pd
import nltk

from nltk.stem import WordNetLemmatizer
from scipy import stats
import textstat
import pickle
import requests
import json
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from textstat import flesch_reading_ease, linsear_write_formula, dale_chall_readability_score


# -- Define helper functions --

# Load target word list and lemmatize
TE_list = pickle.load(open("TE_list.pkl", 'rb'))

def compute_readability(text: str, func):
    try:
        return func(text)
    except Exception:
        return np.nan

def count_sentences(text: str) -> int:
    return textstat.sentence_count(text)

def count_words(text: str) -> int:
    return textstat.lexicon_count(text, removepunct=True)

def avg_reading_time(text: str) -> float:
    # textstat returns time in seconds
    return textstat.reading_time(text)

def lemmatize_word(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word.lower())

lemmatized_word_list = set([lemmatize_word(word) for word in TE_list])

def te_score(text: str) -> float:
    def count_words_not_in_list(sentence):
        words_in_sentence = nltk.word_tokenize(sentence)
        lemmatized_words_in_sentence = [lemmatize_word(word) for word in words_in_sentence]
        count = sum(1 for word in lemmatized_words_in_sentence if word not in lemmatized_word_list)
        return count

    return count_words_not_in_list(text) / count_words(text)

def run_ks_tests(df, score_name, model_name):
    """
    Runs one-tailed KS tests:
        - If it's a "Flesch" type score: alternative='greater'
        - Else: alternative='less'
    """
    if score_name == 'Flesch Reading Ease':
        alternative = 'greater'
    else:
        alternative = 'less'

    phd_data = df[f'{model_name} Graduate School'].dropna()
    hs_data  = df[f'{model_name} High School'].dropna()
    elem_data = df[f'{model_name} Elementary School'].dropna()

    ks_stat_phd_high, p_phd_high = stats.ks_2samp(phd_data, hs_data, alternative=alternative)
    ks_stat_phd_elem, p_phd_elem = stats.ks_2samp(phd_data, elem_data, alternative=alternative)
    ks_stat_high_elem, p_high_elem = stats.ks_2samp(hs_data, elem_data, alternative=alternative)

    print(f"\nOne-Tailed KS Tests ({score_name}): [{alternative}]")
    print(f"HighSchool vs PhD:          KS={ks_stat_phd_high:.4f}, p={p_phd_high:.4f}")
    print(f"Elementary vs PhD:         KS={ks_stat_phd_elem:.4f}, p={p_phd_elem:.4f}")
    print(f"Elementary vs HighSchool:  KS={ks_stat_high_elem:.4f}, p={p_high_elem:.4f}")
    
def classify_explanation(question: str, explanation: str) -> str:
        api_url = "http://127.0.0.1:1234/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        my_prompt = f"""
        You are a helpful assistant. Below is a question followed by an explanation.
        For the explanation, classify whether it is mechanistic or teleological.
        A mechanistic explanation describes how something happens, focusing on the processes, systems, or mechanisms involved.
        A teleological explanation describes something in terms of its goal, purpose, or intended outcome.

        Question: {question}

        Explanation: {explanation}

        Please provide your classification in one word: mechanistic or teleological.
        """
        payload = {
            "model": "llama-3.2-3b-instruct",
            "messages": [{"role": "user", "content": my_prompt}],
            "temperature": 0.1,
            "max_tokens": 10,
            "stream": False
        }
        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            try:
                response = requests.post(api_url, headers=headers, data=json.dumps(payload))
                if response.status_code == 200:
                    try:
                        result = response.json()
                        classification = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                        if "mechanistic" in classification.lower():
                            return "Mechanistic"
                        elif "teleological" in classification.lower():
                            return "Teleological"
                        else:
                            raise ValueError("Invalid classification received.")
                    except (IndexError, KeyError, ValueError, json.JSONDecodeError):
                        retry_count += 1
                else:
                    retry_count += 1
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count == max_retries:
                    return f"Error: {str(e)}"
        return "Error: Failed after 5 attempts."
    
def one_tailed_proportion_test(count1, nobs1, count2, nobs2, alternative="larger"):
    stat, p_value = proportions_ztest(
        count=[count1, count2],
        nobs=[nobs1, nobs2],
        alternative=alternative
    )
    return p_value
    
# Plot the mechanistic proportion bar plot for the three key roles
def compute_prop_and_ci(df, roles_list, label):
    proportions = []
    ci_lower = []
    ci_upper = []
    for role in roles_list:
        total = df[role].shape[0]
        if total == 0:
            proportions.append(0.0)
            ci_lower.append(0.0)
            ci_upper.append(0.0)
        else:
            count_label = (df[role] == label).sum()
            point_est = count_label / total
            low, up = proportion_confint(count_label, total, alpha=0.05, method="wilson")
            proportions.append(point_est * 100.0)
            ci_lower.append(low * 100.0)
            ci_upper.append(up * 100.0)
    return proportions, ci_lower, ci_upper
