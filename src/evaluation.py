import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def cnstrnts_gs(file_path):
    """
    Extracts the constraints from the Gold Standard (GS) .txt file into a dictionary. 
    :param file_path: Path to the GS .txt file. 
    :return: The constraints from the GS in a dictionary. 
    """
    constraints_gs = {}

    with open(file_path, 'r') as file:
        content = file.read()

    for line in content.split('\n'):
        line = line.strip()
        match = re.match(r'^c(\d{1,2}) = (.+)$', line)
        if match:
            current_key = f"c{match.group(1)}"
            constraints_gs[current_key] = match.group(2)

    return constraints_gs

def _preprocess_constraint(constraint):
    """
    Strips the constraint from all unnecesary characters.
    :param constraint: The constraint string in the format "({PROCESS_STEP_1}, {PROCESS_STEP_2}, {directly follows}, {CONSTRAINT})".
    :return: PROCESS_STEP_1 + " " + PROCESS_STEP_2 + " " + CONSTRAINT.
    """
    # Trim leading and trailing whitespace
    constraint = constraint.strip()

    # Check and remove the first two and the last two characters
    if constraint.startswith("({") and constraint.endswith("})"):
        constraint = constraint[2:-2]
    else:
        print("WARNING: Constraint",constraint,"format not right. Check start and end characters (should be '({' and '})').")

    # Split the string at the specified delimiters
    parts = constraint.split("}, {")
    if len(parts) < 4:
        print(f"WARNING: Insufficient parts in constraint {constraint}. Expected at least 4 parts.")

    # Concatenate the relevant parts (PROCESS_STEP_1 + PROCESS_STEP_2 + CONSTRAINT)
    return parts[0] + " " + parts[1] + " " + parts[3]

def sbert_smlarty(first_set, second_set, model='paraphrase-MiniLM-L6-v2'):
    """
    Determines the pairwise similarity scores of two sets of constraints with a pre-trained S-BERT model.
    Preprocesses the constraints to use only PROCESS_STEP_1 + PROCESS_STEP_2 + CONSTRAINT.
    :param first_set: First set of constraints, expected as a dictionary.
    :param second_set: Second set of constraints, expected as a dictionary.
    :param model: Name of the S-BERT model to use. 
    :return: Similartiy scores in a dictionary.
    """

    model = SentenceTransformer(model)
    
    # Preprocess and create embeddings for each set of constraints
    embeddings1 = model.encode([_preprocess_constraint(v) for v in first_set.values()], convert_to_tensor=True)
    embeddings2 = model.encode([_preprocess_constraint(v) for v in second_set.values()], convert_to_tensor=True)

    # Compute cosine similarity between the embeddings
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)

    # Create a dictionary to store similarity scores
    similarity_scores = {}

    # Populate the similarity_scores dictionary
    for i, key1 in enumerate(first_set.keys()):
        for j, key2 in enumerate(second_set.keys()):
            similarity_scores[f'{key1} - {key2}'] = similarity_matrix[i, j]

    return similarity_scores

def _extract_parts(constraint, part):
    """
    Extracts specific parts from a constraint string.
    :param constraint: The constraint string.
    :param part: The part to extract (0 for PROCESS_STEP_1, 1 for PROCESS_STEP_2, 3 for CONSTRAINT).
    :return: Extracted part of the constraint.
    """
    # Trim leading and trailing whitespace
    constraint = constraint.strip()

    # Check and remove the first two and the last two characters
    if constraint.startswith("({") and constraint.endswith("})"):
        constraint = constraint[2:-2]
    else:
        print("WARNING: Constraint",constraint,"format not right. Check start and end characters (should be '({' and '})').")

    # Split the string at the specified delimiters
    parts = constraint.split("}, {")
    if len(parts) < 4:
        print(f"WARNING: Insufficient parts in constraint {constraint}. Expected at least 4 parts.")

    # Return the requested part
    return parts[part]

# def sbert_smlarty_cmpntns(similarity_scores_dict, first_set, second_set, threshold=0.5, model='paraphrase-MiniLM-L6-v2', matching_mode='unique'):
#     """
#     Analyzes similarity scores to find matches and organize them into DataFrames.
#     :param similarity_scores_dict: Dictionary containing similarity scores between constraints.
#     :param first_set: First set of constraints as a dictionary.
#     :param second_set: Second set of constraints as a dictionary.
#     :param threshold: Minimum similarity score to consider a match. Default is 0.5.
#     :param model: Name of the S-BERT model to use. 
#     :param matching_mode: Mode of matching, either 'unique' or 'multiple'. Default is 'unique'.
#     :return: Three DataFrames with matching constraints details.
#     """
    
#     # Initialize S-BERT model
#     model = SentenceTransformer(model)

#     # Initialize DataFrames
#     cols = ["Constraint pair", "Group similarity", "From", "To", "Similarity"]
#     matches_step_1 = pd.DataFrame(columns=cols)
#     matches_step_2 = pd.DataFrame(columns=cols)
#     matches_constraints = pd.DataFrame(columns=cols)

#     # Process each constraint in the second set
#     for key2, constraint2 in second_set.items():
#         matched_key1 = None
#         max_similarity = threshold
#         print("key2:", key2)

#         # Find the best match above threshold
#         for key_pair, similarity in similarity_scores_dict.items():
#             key1, _ = key_pair.split(' - ')
#             if key2 in key_pair and key1 in first_set.keys() and similarity > max_similarity:
#                 matched_key1 = key1
#                 max_similarity = similarity

#         if matched_key1:
#             print("matched_key1:", matched_key1)
#             # Placeholder for the extract_parts() function
#             # Extract parts of the constraints
#             from_step_1 = _extract_parts(first_set[matched_key1], 0)
#             to_step_1 = _extract_parts(constraint2, 0)
#             from_step_2 = _extract_parts(first_set[matched_key1], 1)
#             to_step_2 = _extract_parts(constraint2, 1)
#             from_constraint = _extract_parts(first_set[matched_key1], 3)
#             to_constraint = _extract_parts(constraint2, 3)

#             # Calculate S-BERT similarity for each part
#             similarity_step_1 = cosine_similarity(model.encode([from_step_1]), model.encode([to_step_1]))[0][0]
#             similarity_step_2 = cosine_similarity(model.encode([from_step_2]), model.encode([to_step_2]))[0][0]
#             similarity_constraint = cosine_similarity(model.encode([from_constraint]), model.encode([to_constraint]))[0][0]

#             # Add to DataFrames
#             matches_step_1.loc[len(matches_step_1)] = [key_pair, max_similarity, from_step_1, to_step_1, similarity_step_1]
#             matches_step_2.loc[len(matches_step_2)] = [key_pair, max_similarity, from_step_2, to_step_2, similarity_step_2]
#             matches_constraints.loc[len(matches_constraints)] = [key_pair, max_similarity, from_constraint, to_constraint, similarity_constraint]

#             # In 'unique' mode, remove processed constraints
#             if matching_mode == 'unique':
#                 first_set.pop(matched_key1, None)
#                 if not first_set:
#                     break

#     return matches_step_1, matches_step_2, matches_constraints

def sbert_smlarty_cmpntns(similarity_scores_dict, first_set, second_set, threshold=0.5, model='paraphrase-MiniLM-L6-v2', matching_mode='unique'):
    """
    Analyzes similarity scores to find matches and organize them into DataFrames.
    :param similarity_scores_dict: Dictionary containing similarity scores between constraints.
    :param first_set: First set of constraints as a dictionary.
    :param second_set: Second set of constraints as a dictionary.
    :param threshold: Minimum similarity score to consider a match. Default is 0.5.
    :param model: Name of the S-BERT model to use. 
    :param matching_mode: Mode of matching, either 'unique' or 'multiple'. Default is 'unique'.
    :return: Three DataFrames with matching constraints details.
    """
    
    # Initialize S-BERT model
    model = SentenceTransformer(model)

    # Initialize DataFrames
    cols = ["Constraint pair", "Group similarity", "From", "To", "Similarity"]
    matches_step_1 = pd.DataFrame(columns=cols)
    matches_step_2 = pd.DataFrame(columns=cols)
    matches_constraints = pd.DataFrame(columns=cols)

    # Sort and filter similarity_scores_dict
    sorted_scores = sorted([(k, v) for k, v in similarity_scores_dict.items() if v >= threshold], key=lambda x: x[1], reverse=True)

    # Process each entry in the sorted and filtered list
    processed_keys_set1 = set()
    processed_keys_set2 = set()
    for key_pair, group_similarity in sorted_scores:
        key1, key2 = key_pair.split(' - ')
        
        # In 'unique' mode, skip if any key has already been processed
        if matching_mode == 'unique':
            if key1 in processed_keys_set1 or key2 in processed_keys_set2:
                continue

        # Extract parts of the constraints
        from_step_1 = _extract_parts(first_set[key1], 0)
        to_step_1 = _extract_parts(second_set[key2], 0)
        from_step_2 = _extract_parts(first_set[key1], 1)
        to_step_2 = _extract_parts(second_set[key2], 1)
        from_constraint = _extract_parts(first_set[key1], 3)
        to_constraint = _extract_parts(second_set[key2], 3)

        # Calculate S-BERT similarity for each part
        similarity_step_1 = cosine_similarity(model.encode([from_step_1]), model.encode([to_step_1]))[0][0]
        similarity_step_2 = cosine_similarity(model.encode([from_step_2]), model.encode([to_step_2]))[0][0]
        similarity_constraint = cosine_similarity(model.encode([from_constraint]), model.encode([to_constraint]))[0][0]

        # Add to DataFrames
        matches_step_1.loc[len(matches_step_1)] = [key_pair, group_similarity, from_step_1, to_step_1, similarity_step_1]
        matches_step_2.loc[len(matches_step_2)] = [key_pair, group_similarity, from_step_2, to_step_2, similarity_step_2]
        matches_constraints.loc[len(matches_constraints)] = [key_pair, group_similarity, from_constraint, to_constraint, similarity_constraint]

        # Mark keys as processed
        processed_keys_set1.add(key1)
        processed_keys_set2.add(key2)

    return matches_step_1, matches_step_2, matches_constraints


