import re
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

def sbert_smlarty(constraints1, constraints2, model='paraphrase-MiniLM-L6-v2'):
    """
    Determines the pairwise similarity scores of two sets of constraints with a pre-trained S-BERT model.
    :param constraints1: First set of constraints, expected as a dictionary.
    :param constraints2: Second set of constraints, expected as a dictionary.
    :param model: Name of the S-BERT model to use. For more information, see https://www.sbert.net/docs/pretrained_models.html#model-overview. 
    """
    model = SentenceTransformer(model)
    
    # Create embeddings for each set of constraints
    embeddings1 = model.encode(list(constraints1.values()), convert_to_tensor=True)
    embeddings2 = model.encode(list(constraints2.values()), convert_to_tensor=True)

    # Compute cosine similarity between the embeddings
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)

    # Create a dictionary to store similarity scores
    similarity_scores = {}

    # Populate the similarity_scores dictionary
    for i, key1 in enumerate(constraints1.keys()):
        for j, key2 in enumerate(constraints2.keys()):
            similarity_scores[f'{key1} - {key2}'] = similarity_matrix[i, j]

    return similarity_scores
