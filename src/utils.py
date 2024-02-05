import os
import docx2txt
import sys
import subprocess
import spacy


def load_spacy(model):
    """
    Evaluating if the pre-trained SpaCy model is already installed, and installing it, if it isn't. 

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

# TODO: Generalize
def docx_to_txt(docx_path='EB115.docx',txt_path='output.txt'):
    """
    Convert .docx to .txt.
    :param docx_path: Path of the .doxc file.
    :param txt_path: Path of the .txt file.
    """
    MY_TEXT = docx2txt.process(docx_path)
    with open(txt_path, "w") as text_file:
        print(MY_TEXT, file=text_file)

# docx_to_txt(docx_path='input_cdm_03.docx')