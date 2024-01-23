import docx2txt
import sys
import subprocess
import spacy


def ld_spcy_mdl(model):
    """
    Evaluating if the pre-trained SpaCy model is already installed, and installing it, if it isn't. 

    :param model: Name of the pre-trained SpaCy  model. 
    :return: Pre-trained SpaCy model.
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

def prnt_brk(text, linebreak=60):
    """
    Break the text into lines before printing.
    :param text: Input to print.
    :param linebreak: Number of chars after which the line breaks.
    """
# Split the text into lines
    lines = [text[i:i+linebreak] for i in range(0, len(text), linebreak)]

    # Print each line
    for line in lines:
        print(line)

# docx_to_txt(docx_path='input_cdm_03.docx')