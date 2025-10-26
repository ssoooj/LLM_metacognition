import pandas as pd
import numpy as np
import ast
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Set, Any

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

try:
    print("[Start] Sentence-BERT loaded")
    SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("[Done] Sentence-BERT loaded\n")
except Exception as e:
    print(f"Error: {e}")
    SBERT_MODEL = None

SIMILARITY_THRESHOLD = 0.85

CONSORT_ITEMS_DESC: Dict[str, str] = {
    '1a': 'Identification as a randomized trial in the title',
    '2a': 'Scientific background and explanation of rationale',
    '3a': 'Description of trial design (e.g., parallel, factorial) including allocation ratio',
    '4a': 'Eligibility criteria for participants',
    '4b': 'Settings and locations where the data were collected',
    '5': 'The interventions for each group with sufficient details to allow replication',
    '6a': 'Completely defined pre-specified primary and secondary outcome measures',
    '7a': 'How sample size was determined',
    '7b': 'When applicable, explanation of any interim analyses and stopping guidelines',
    '8a': 'Method used to generate the random allocation sequence',
    '8b': 'Type of randomisation; details of any restriction',
    '9': 'Mechanism used to implement the random allocation sequence',
    '10': 'Who generated the random allocation sequence, who enrolled participants, and who assigned participants to interventions',
    '11a': 'If done, who was blinded after assignment to interventions and how',
    '12a': 'Statistical methods used to compare groups for primary and secondary outcomes',
    '15': 'For each group, the numbers of participants who were randomly assigned, received intended treatment, and were analysed for the primary outcome',
    '23': 'Registration number and name of trial registry'
}

def parse_str_list(s: Any) -> List[str]:
    if not isinstance(s, str) or not (s.startswith('[') and s.endswith(']')):
        return []
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

def has_evidence(sentences_str: Any) -> int:
    if not isinstance(sentences_str, str) or sentences_str.strip() == '':
        return 0
    try:
        sentences_list = ast.literal_eval(sentences_str)
        return 1 if isinstance(sentences_list, list) and len(sentences_list) > 0 else 0
    except (ValueError, SyntaxError):
        return 0

def prepare_data_for_analysis(exp_file: str, methods_file: str) -> pd.DataFrame:
    try:
        df_exp = pd.read_csv(exp_file, dtype=str)

        evidence_score_map = {'strong': 1.0, 'moderate': 0.5, 'weak': 0.1, 'none': 0.0}
        df_exp['evidence_score'] = df_exp['Evidence_Strength'].map(evidence_score_map).fillna(0)
        df_exp['has_evidence'] = df_exp['Extracted_Sentences'].apply(has_evidence)
        df_exp['Confidence'] = pd.to_numeric(df_exp['Confidence'], errors='coerce').fillna(0)
        df_exp['row_score'] = (df_exp['Confidence'] / 100) * df_exp['evidence_score'] * df_exp['has_evidence']

        df_exp['ai_sentences'] = df_exp['Extracted_Sentences'].apply(parse_str_list)
        df_exp['CONSORT_Item'] = df_exp['CONSORT_Item'].apply(parse_str_list)
        df_exp = df_exp.explode('CONSORT_Item').dropna(subset=['CONSORT_Item'])
        df_exp['CONSORT_Item'] = df_exp['CONSORT_Item'].str.strip()

    except FileNotFoundError:
        print(f"   Couldn't find '{exp_file}'")
        return pd.DataFrame()

    try:
        df_methods = pd.read_csv(methods_file, dtype=str)
        df_methods['CONSORT_Item'] = df_methods['CONSORT_Item'].apply(parse_str_list)
        df_methods = df_methods.explode('CONSORT_Item').dropna(subset=['CONSORT_Item'])
        df_methods['CONSORT_Item'] = df_methods['CONSORT_Item'].str.strip()
        df_methods = df_methods[df_methods['CONSORT_Item'] != '0']
    except FileNotFoundError:
        print(f"   Couldn't find '{methods_file}'")
        return pd.DataFrame()

    ground_truth = df_methods.groupby(['PMCID', 'CONSORT_Item'])['text'].apply(list).reset_index()
    ground_truth.rename(columns={'text': 'expert_sentences'}, inplace=True)

    df_exp['PMCID'] = df_exp['PMCID'].str.strip()
    ground_truth['PMCID'] = ground_truth['PMCID'].str.strip()
    
    df_merged = pd.merge(df_exp, ground_truth, on=['PMCID', 'CONSORT_Item'], how='inner')

    if df_merged.empty:
        return pd.DataFrame()

    df_merged['item_description'] = df_merged['CONSORT_Item'].map(CONSORT_ITEMS_DESC)

    return df_merged

def calculate_semantic_f1(ai_sents: List[str], expert_sents: List[str]) -> tuple[float, float, float]:
    if not ai_sents or not expert_sents or not SBERT_MODEL:
        return 0.0, 0.0, 0.0

    try:
        ai_embeddings = SBERT_MODEL.encode(ai_sents)
        expert_embeddings = SBERT_MODEL.encode(expert_sents)
        sim_matrix = cosine_similarity(expert_embeddings, ai_embeddings)

        true_positives = 0
        matched_ai_indices: Set[int] = set()

        for i in range(len(expert_sents)):
            if sim_matrix.shape[1] == 0: continue
            
            best_match_idx, max_sim = -1, -1.0
            for j in range(len(ai_sents)):
                if j not in matched_ai_indices and sim_matrix[i, j] > max_sim:
                    max_sim, best_match_idx = sim_matrix[i, j], j
            
            if max_sim >= SIMILARITY_THRESHOLD and best_match_idx != -1:
                true_positives += 1
                matched_ai_indices.add(best_match_idx)

        precision = true_positives / len(ai_sents) if len(ai_sents) > 0 else 0
        recall = true_positives / len(expert_sents) if len(expert_sents) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1_score
    except Exception as e:
        print(f"Error: {e}")
        return 0.0, 0.0, 0.0

def calculate_reasoning_score(reasoning: str, has_extracted: bool, item_desc: str) -> int:
    reasoning_lower = str(reasoning).lower()
    item_desc_lower = str(item_desc).lower()
    
    no_extraction_keywords = ["no", "not found", "empty", "missing", "not mention", "impossible", "cannot find", "absent", "n/a"]
    reasoning_indicates_no_extraction = any(kw in reasoning_lower for kw in no_extraction_keywords)
    consistency_score = 1 if has_extracted != reasoning_indicates_no_extraction else 0

    item_keywords = {kw for kw in item_desc_lower.replace('(e.g.,', '').replace(')', '').split() if len(kw) > 3}
    rationale_score = 1 if item_keywords and any(kw in reasoning_lower for kw in item_keywords) else 0
    
    return consistency_score + rationale_score

def calculate_all_metrics(row: pd.Series) -> pd.Series:
    ai_sents = row.get('ai_sentences', [])
    expert_sents = row.get('expert_sentences', [])
    
    precision, recall, f1_score = calculate_semantic_f1(ai_sents, expert_sents)
    
    reasoning_score = calculate_reasoning_score(
        reasoning=row.get('Reasoning', ""),
        has_extracted=bool(ai_sents),
        item_desc=row.get('item_description', "")
    )

    return pd.Series([precision, recall, f1_score, reasoning_score])

def main_analysis(exp_file: str, methods_file: str):
    if not SBERT_MODEL:
        print("\nSBERT model not loaded. Stopping.")
        return

    df_analysis = prepare_data_for_analysis(exp_file, methods_file)
    
    if df_analysis.empty:
        print("\nNo data to be processed.")
        return

    metric_results = df_analysis.apply(calculate_all_metrics, axis=1, result_type='expand')
    metric_results.columns = ['Semantic_Precision', 'Semantic_Recall', 'Semantic_F1', 'Reasoning_Score']
    df_analysis = pd.concat([df_analysis.reset_index(drop=True), metric_results], axis=1)
    
    output_details_file = 'analysis_details_integrated.csv'
    df_analysis.to_csv(output_details_file, index=False, encoding='utf-8-sig')
    
    final_scores = df_analysis.groupby(['Model', 'Prompt_Template']).agg(
        Mean_F1_Score=('Semantic_F1', 'mean'),
        Mean_Reasoning_Score=('Reasoning_Score', 'mean'),
        Mean_Compliance_Score=('row_score', 'mean')
    ).sort_values(by=['Mean_F1_Score', 'Mean_Compliance_Score'], ascending=[False, False])
    
    final_scores['Normalized_Compliance_Score'] = final_scores['Mean_Compliance_Score'] * 100
    
    final_scores = final_scores[['Normalized_Compliance_Score', 'Mean_F1_Score', 'Mean_Reasoning_Score']]
    
    print(final_scores.to_string(float_format="%.4f"))
    
    output_summary_file = 'final_performance_summary_integrated.csv'
    final_scores.to_csv(output_summary_file, encoding='utf-8-sig')

if __name__ == "__main__":
    experiment_file_path = '/Users/sohyeon/Downloads/VitalLab/LLM_Cognitive_Abilities/data/experiment_results_medgemma.csv'
    methods_file_path = '/Users/sohyeon/consort-tm/Methods_all.csv'
    
    main_analysis(exp_file=experiment_file_path, methods_file=methods_file_path)