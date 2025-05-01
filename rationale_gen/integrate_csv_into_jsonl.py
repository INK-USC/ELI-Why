#!/usr/bin/env python3
import csv
import json
import os
import re

def remove_chinese(text):
    """
    Remove Chinese characters and common Chinese punctuation from the input text.
    The regex now covers:
      - Chinese ideographs: \u4e00-\u9fff
      - CJK symbols and punctuation: \u3000-\u303f
      - Fullwidth forms: \uff00-\uffef
      - Common quotation marks: “ (U+201C) and ” (U+201D)
    """
    pattern = r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u201c\u201d]+'
    return re.sub(pattern, '', text)

def remove_chinese_from_record(record):
    """
    Iterate over all keys in a record and remove Chinese characters and punctuation from string values.
    """
    for key, value in record.items():
        if isinstance(value, str):
            record[key] = remove_chinese(value)
    return record

def load_csv(csv_path):
    """
    Loads Qwen2.5 rationales from a CSV file.
    Returns a dictionary keyed on (Question, Domain, Discipline)
    where each value is the CSV row as a dictionary.
    """
    mapping = {}
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = (
                row.get("Question", "").strip(),
                row.get("Domain", "").strip(),
                row.get("Discipline", "").strip()
            )
            mapping[key] = row
    return mapping

def integrate_jsonl(jsonl_path, integration_groups, output_path):
    """
    Reads the JSONL file and integrates the rationales from CSV files into new fields.
    For each integration group, adds fields using keys:
      "{prefix} Graduate School", "{prefix} High School",
      "{prefix} Elementary School", and "{prefix} Default".
    Then filters out Chinese characters and punctuation from all string fields.
    The updated records are written to a new JSONL file.
    """
    with open(jsonl_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (
                record.get("Question", "").strip(),
                record.get("Domain", "").strip(),
                record.get("Discipline", "").strip()
            )
            for group in integration_groups:
                prefix = group["prefix"]
                mapping = group["mapping"]
                fields = group["fields"]
                if key in mapping:
                    csv_row = mapping[key]
                    for field in fields:
                        record[f"{prefix} {field}"] = csv_row.get(field, "")
                else:
                    for field in fields:
                        record[f"{prefix} {field}"] = ""
            # Remove Chinese characters and punctuation from all string fields.
            record = remove_chinese_from_record(record)
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    # Define file paths and integration groups.
    jsonl_path = os.path.join("..", "ELI_Why_with_rationales.jsonl")
    output_path = "integrated_ELI_Why_with_rationales.jsonl"
    
    integration_groups = [
        {
            "prefix": "Qwen2.5",
            "csv_path": "Qwen2.5_rationales.csv",
            "fields": ["Graduate School", "High School", "Elementary School", "Default"]
        },
        {
            "prefix": "Gemma3",
            "csv_path": "Gemma3_rationales.csv",
            "fields": ["Graduate School", "High School", "Elementary School", "Default"]
        },
        {
            "prefix": "R1_Distilled_Llama",
            "csv_path": "R1_Distilled_Llama_rationales.csv",
            "fields": ["Graduate School", "High School", "Elementary School", "Default"]
        }
    ]
    
    # Load CSV mappings for all integration groups.
    for group in integration_groups:
        group["mapping"] = load_csv(group["csv_path"])
    
    integrate_jsonl(jsonl_path, integration_groups, output_path)
    print(f"Integration complete. Output written to {output_path}")

if __name__ == "__main__":
    main()
