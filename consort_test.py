import pandas as pd
import json
import time
import os
import requests
import csv
import ast
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import openai

load_dotenv()
OPENAI_API_BASE = "http://localhost:1234/v1"
OPENAI_API_KEY = "lm-studio"
MAX_PAPER_LENGTH_CHARS = 10000
RESULTS_FILE = 'experiment_results.csv'
METHODS_FILE = 'Methods_all.csv'

class PaperFetcher:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.api_key = os.getenv('NCBI_API_KEY')
        self.cache_dir = "paper_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        if not self.api_key:
            raise ValueError("Can't find NCBI_API_KEY in the .env file.")

    def get_full_text(self, pmcid: str) -> str:
        cache_path = os.path.join(self.cache_dir, f"{pmcid}.txt")
        if os.path.exists(cache_path):
            print(f"    - From local cache {pmcid} loading...")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        print(f"    - Downloadig through NCBI API {pmcid}...")
        xml_content = self._fetch_xml_content(pmcid)
        if not xml_content:
            return f"[Error fetching text for {pmcid}]"
            
        full_text = self._extract_text_from_xml(xml_content)
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
            
        return full_text

    def _fetch_xml_content(self, pmcid: str) -> str | None:
        numeric_id = pmcid.replace('PMC', '')
        fetch_url = (f"{self.base_url}efetch.fcgi?"
                     f"db=pmc&id={numeric_id}"
                     f"&rettype=xml&api_key={self.api_key}")
        try:
            response = requests.get(fetch_url, timeout=60)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"      - Error fetching {pmcid}: {e}")
            return None

    def _extract_text_from_xml(self, xml_content: str) -> str:
        soup = BeautifulSoup(xml_content, 'lxml-xml')
        abstract_text = "\n".join([p.get_text(strip=True) for p in soup.find('abstract').find_all('p')]) if soup.find('abstract') else ""
        body_text = "\n".join([p.get_text(strip=True) for p in soup.find('body').find_all('p')]) if soup.find('body') else ""
        return f"ABSTRACT:\n{abstract_text}\n\nBODY:\n{body_text}".strip()

def call_llm(prompt: str, model_name: str) -> dict:
    print(f"--- Calling model: {model_name} via LM Studio ---")
    try:
        llm = ChatOpenAI(model=model_name, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE, temperature=0.0, max_tokens=2048, request_timeout=300)
        response = llm.invoke([HumanMessage(content=prompt)])
        response_content = response.content
        
        json_block_start = response_content.find('''```json''')
        if json_block_start != -1:
            json_start = response_content.find('{', json_block_start)
            json_end = response_content.rfind('}') + 1
            json_string = response_content[json_start:json_end]
        else:
            json_string = response_content
            
        output = json.loads(json_string)
        if not isinstance(output, dict): raise json.JSONDecodeError("Output is not a dict", "", 0)
            
    except json.JSONDecodeError:
        output = {"reasoning": response_content, "extracted_sentences": []}
    except Exception as e:
        output = {"reasoning": f"ERROR: {str(e)}", "extracted_sentences": []}
    
    time.sleep(15)
    return output

CONSORT_ITEMS_17 = {
    '1a': 'Identification as a randomized trial in the title', '2a': 'Scientific background and explanation of rationale',
    '3a': 'Description of trial design (e.g., parallel, factorial) including allocation ratio', '4a': 'Eligibility criteria for participants',
    '4b': 'Settings and locations where the data were collected', '5': 'The interventions for each group with sufficient details to allow replication',
    '6a': 'Completely defined pre-specified primary and secondary outcome measures', '7a': 'How sample size was determined',
    '7b': 'When applicable, explanation of any interim analyses and stopping guidelines', '8a': 'Method used to generate the random allocation sequence',
    '8b': 'Type of randomisation; details of any restriction', '9': 'Mechanism used to implement the random allocation sequence',
    '10': 'Who generated the random allocation sequence, who enrolled participants, and who assigned participants to interventions', '11a': 'If done, who was blinded after assignment to interventions and how',
    '12a': 'Statistical methods used to compare groups for primary and secondary outcomes', '15': 'For each group, the numbers of participants who were randomly assigned, received intended treatment, and were analysed for the primary outcome',
    '23': 'Registration number and name of trial registry'
}
MODELS_TO_TEST = ['medgemma-27b-text-it-mlx']
PROMPT_TEMPLATES = {
    'Zero-Shot CoT': """You are an expert clinical research auditor. 
    Your task is to analyze a research paper and extract sentences corresponding to a specific CONSORT item.

    **Paper to Audit:**
    - PMCID: {pmcid}
    - Full Text:
    ---
    {paper_text}
    ---

    **Your Objective:**
    - Find evidence for: **CONSORT Item {item_number}: '{item_description}'**.

    **Instructions:**
    1.  **Think Step-by-Step**: First, write down your reasoning process for identifying the relevant sentences in a dedicated "reasoning" field. Explain which parts of the text you searched and why you selected (or did not select) certain sentences.
    2.  **Extract Sentences**: Based on your reasoning, extract the exact, verbatim sentences.
    3.  **Final Answer**: Provide your answer as a single, valid JSON object with two keys: "reasoning" (your thought process) and "extracted_sentences" (a list of strings). If no sentences are found, "extracted_sentences" must be an empty list.

    Let's think step by step.
    
    **JSON Output Format:**
    {{
      "reasoning": "...",
      "extracted_sentences": ["...", "..."]
    }}
    """,
    'Role-Playing': """Imagine you are a meticulous auditor for a top-tier medical journal. Your reputation for precision is unparalleled.

    You have been assigned a research paper for audit.
    - **Paper PMCID**: {pmcid}
    - **Full Text**:
    ---
    {paper_text}
    ---

    Your current objective is to verify compliance with a single, specific guideline:
    - **CONSORT Item {item_number}**: '{item_description}'

    **Your Required Deliverable:**
    You must produce a structured JSON report. This is not optional.
    1.  First, in the `reasoning` field, articulate your expert thought process. Detail which sections you examined (e.g., Methods, Results) and justify your decision to include or exclude sentences.
    2.  Second, in the `extracted_sentences` field, provide a list of the exact, verbatim sentences that directly and explicitly satisfy the item's requirement.
    3.  If no sentences in the paper meet the standard, you must state this in your reasoning and return an empty list for `extracted_sentences`.

    Your final output must be only the JSON object, without any additional commentary.

    **JSON Output Format:**
    {{
      "reasoning": "I will search the Methods and Results sections for specific keywords related to the intervention, such as drug dosages and administration schedules, to identify the sentences describing the intervention in sufficient detail for replication.",
      "extracted_sentences": ["...", "..."]
    }}
    """,
    'Few-Shot': """You are an expert clinical research auditor. Your task is to identify sentences in a research paper that correspond to a specific CONSORT guideline item and explain your reasoning. Follow the format of the examples provided.

    --- EXAMPLES ---

    [Example 1]
    - CONSORT Item: '3a: Description of trial design (e.g., parallel, factorial) including allocation ratio'
    - Sentence from paper: "This was a multicenter, randomized, double-blind, parallel-group study."
    - Your JSON Output:
    {{
      "reasoning": "The sentence explicitly uses keywords like 'randomized', 'double-blind', and 'parallel-group', which directly describe the trial design as required by item 3a.",
      "extracted_sentences": [
        "This was a multicenter, randomized, double-blind, parallel-group study."
      ]
    }}

    [Example 2]
    - CONSORT Item: '7a: How sample size was determined'
    - Sentence from paper: "A sample size of 150 patients per group was calculated to provide 80% power to detect a significant difference."
    - Your JSON Output:
    {{
      "reasoning": "This sentence directly states the calculated sample size and the statistical power (80%), which is the core information needed for item 7a.",
      "extracted_sentences": [
        "A sample size of 150 patients per group was calculated to provide 80% power to detect a significant difference."
      ]
    }}

    --- END OF EXAMPLES ---

    Now, it is your turn.

    **Paper to Audit:**
    - PMCID: {pmcid}
    - Full Text:
    ---
    {paper_text}
    ---

    **Your Objective:**
    - Find evidence for: **CONSORT Item {item_number}: '{item_description}'**.

    Provide your answer as a single, valid JSON object with the keys "reasoning" and "extracted_sentences", following the format in the examples. If no relevant sentences are found, state this in your reasoning and return an empty list.
    """
}

def run_experiment():
    print("Start Experiment")
    
    fetcher = PaperFetcher()
    
    try:
        df_source = pd.read_csv(METHODS_FILE, dtype=str)
    except FileNotFoundError:
        print(f"Error: Can't find the file '{METHODS_FILE}'. Stopping.")
        return

    def clean_consort_item(item_str: str) -> str:
        try:
            items = ast.literal_eval(item_str)
            if isinstance(items, list) and items: return str(items).strip()
        except: return str(item_str).strip()
        return None

    df_source['CONSORT_Item'] = df_source['CONSORT_Item'].apply(clean_consort_item)
    tasks_with_ground_truth = df_source[df_source['CONSORT_Item'] != '0'].dropna(subset=['CONSORT_Item'])
    
    tasks_to_run = tasks_with_ground_truth[['PMCID', 'CONSORT_Item']].drop_duplicates().values.tolist()
    print(f"Conducting an experiment for total {len(tasks_to_run)} [paper-item] combinations")

    completed_tasks = set()
    file_exists = os.path.exists(RESULTS_FILE)
    if file_exists:
        print(f"Found existing '{RESULTS_FILE}'. Skipping that's been already done.")
        try:
            df_results = pd.read_csv(RESULTS_FILE, dtype=str)
            for _, row in df_results.iterrows():
                completed_tasks.add((row['PMCID'], row['CONSORT_Item'], row['Model'], row['Prompt_Template']))
        except Exception as e:
            print(f"  - Waring: ({e}).")

    with open(RESULTS_FILE, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(RESULTS_FILE) == 0:
            writer.writerow(["PMCID", "CONSORT_Item", "Model", "Prompt_Template", "Reasoning", "Extracted_Sentences"])

        total_tasks_to_run = len(tasks_to_run) * len(MODELS_TO_TEST) * len(PROMPT_TEMPLATES)
        pending_tasks = total_tasks_to_run - len(completed_tasks)
        current_task_num = 0
        print(f"Total {total_tasks_to_run} tasks on track, {len(completed_tasks)} done. {pending_tasks} left.")

        for pmcid, item_number in tasks_to_run:
            item_description = CONSORT_ITEMS_17.get(item_number, "Unknown Item")
            
            for model_name in MODELS_TO_TEST:
                for template_name, template_text in PROMPT_TEMPLATES.items():
                    task_id = (str(pmcid), str(item_number), str(model_name), str(template_name))
                    if task_id in completed_tasks:
                        continue

                    current_task_num += 1
                    print(f"\n[{current_task_num}/{pending_tasks}] Processing -> PMCID: {pmcid}, Item: {item_number}")
                    
                    paper_text = fetcher.get_full_text(pmcid)
                    if "[Error" in paper_text:
                        print(f"    - {pmcid} failed to crawl text.")
                        break # next paper

                    prompt = template_text.format(
                        pmcid=pmcid, paper_text=paper_text[:MAX_PAPER_LENGTH_CHARS],
                        item_number=item_number, item_description=item_description
                    )
                    
                    ai_response = call_llm(prompt, model_name)
                    
                    reasoning = ai_response.get("reasoning", "")
                    sentences = json.dumps(ai_response.get("extracted_sentences", []), ensure_ascii=False)
                    
                    writer.writerow([pmcid, item_number, model_name, template_name, reasoning, sentences])
                    f.flush()
                    
                    completed_tasks.add(task_id)

    print(f"\nDone! Saved to '{RESULTS_FILE}' file.")

if __name__ == "__main__":
    run_experiment()