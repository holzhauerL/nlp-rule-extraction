import os
import sys
import subprocess
import spacy
import matplotlib.colors as mcolors
from sentence_transformers import SentenceTransformer

def load_spacy(model):
    """
    Evaluating if the pre-trained spaCy model is already installed, and installing it, if it isn't. 

    :param model: Name of the pre-trained spaCy  model. 
    :return: Pre-trained spaCy model.
    """
    try:
        # Try to load the spaCy model
        nlp = spacy.load(model)
        print('The model', model, 'is already installed!')
    except OSError:
        # If loading fails, install the model using subprocess
        print("Installing", model, "model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
    return nlp

def load_sbert(model):
    """
    Loading the S-BERT model.

    :param model: Name of the pre-trained S-BERT model. 
    :return: Pre-trained S-BERT model.
    """
    return SentenceTransformer(model)

def generate_paths(cases, mode):
    """
    Generating the paths for the input and output files for all the cases.

    :param cases: Dictionary with the cases. The keys are the names of the cases, the values can be a string (if the name of the folder and the file suffix are the same) or a tuple (if not).
    :param mode: Determines if 'input' or 'output' files should be considered.
    :return: Dictionary with file paths for all the use cases.
    """
    prefix = 'input_' if mode == 'input' else 'output_' if mode == 'output' else None
    if prefix is None:
        print("Unknown mode, no file paths generated")
        return None

    return {key: os.path.join('..', 'data', value[0] if isinstance(value, tuple) else value, prefix + (value[1] if isinstance(value, tuple) else value) + ".txt") for key, value in cases.items()}

def generate_color_mapping(cases):
    """
    Generate a color mapping for each unique case name.

    :param cases: A dictionary containing case names as keys.
    :return: A dictionary mapping each case name to a color string.
    """
    color_mapping = {}
    available_colors = list(mcolors.TABLEAU_COLORS.keys())
    
    # List of predefined colors in a specific order
    predefined_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    for case in cases.keys():
        if case not in color_mapping:
            if predefined_colors:
                color = predefined_colors.pop(0)
            else:
                # If predefined colors are exhausted, use random colors
                color = available_colors.pop()
            color_mapping[case] = color

    return color_mapping