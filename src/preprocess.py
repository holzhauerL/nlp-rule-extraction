import os
import sys
import pickle
from termcolor import colored
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import numerizer

def txt_to_df(file_path, section_start="# ", subsection_start="## "):
    """
    Reads the text from a .txt file in a certain format, breaks it down into lines and returns it as a pandas Dataframe, organised into sections and subsections. 
    :param file_path: Relative path to the .txt input file.
    :param section_start: String indicating the start of a section title.
    :param subsection_start: String indicating the start of a subsection title.
    :return: Dataframe with the columns "Section", "Raw" and "Processed" as well as {current_section}/{current_subsection} as a syntax for concatinating sections and sub-sections. 
    """
    raw = open(file_path).read()

    lines = raw.split('\n')
    columns = []

    current_section = ''
    current_subsection = ''

    for line in lines:
        if line.startswith(section_start):
            current_section = line[2:].strip()
            current_subsection = ''
        elif line.startswith(subsection_start):
            current_subsection = line[3:].strip()
        else:
            if current_subsection:
                column_name = f"{current_section}/{current_subsection}"
            else:
                column_name = current_section

            if not columns or column_name != columns[-1][0]:
                columns.append((column_name, []))

            columns[-1][1].append(line)
    
    data = {"Section": [], "Raw": [], "Processed": []}
    for column_name, column_content in columns:
        data["Section"].append(column_name)
        data["Raw"].append('\n'.join(column_content))
        data["Processed"].append('')

    return pd.DataFrame(data)

def rmv_and_rplc(text, remove=["\n", "\t"], replace={"->": "leads to", "-->": "leads to"}, space=True):
    """
    Removes and replaces literals in the text . 
    :param text: Input string.
    :param remove: List of strings to remove.
    :param replace: Mapping of strings to replace.
    :param space: Flag to choose if a space should be used for the replacements.
    :return: Output string with replacements and removals. 
    """

    # Apply replacements
    for key, value in replace.items():
        text = text.replace(key, value)

    # Remove unwanted strings
    for item in remove:
        if space:
            text = text.replace(item, " ")
        else:
            text = text.replace(item, "")

    return text

def chng_stpwrds(add=[],remove=[],remove_numbers=False,restore_default=False, default_filename = 'stopwords_en.pickle',verbose=False, model='en_core_web_lg'):
    """
    Adds and removes stop words to the default of spacy. 
    :param add: Stop words to add, e.g. ["by", "the"].
    :param remove: Stop words to remove, e.g. ["to", "one"].
    :param remove_numbers: Flag to remove all numbers in natural language from the stop words list.
    :param restore_default: Flag to restore the default stop word selection provided by spacy. 
    :param default_filename: Name of the file with the default stop words. Only relevant if restore_default == True.
    :param verbose: If True, stop words are printed.
    :param model: Pretrained spacy pipeline to use.
    :return: Stop words.
    """

    if restore_default:
        default_filepath = os.path.join('..', 'src', default_filename) # works in src and notebooks folder
        with open(default_filepath, 'rb') as file:
            cls = pickle.load(file)
    else:
        if remove_numbers:
            stpwrds = sorted(STOP_WORDS)
            nlp = spacy.load(model)

            for item in stpwrds:
                doc = nlp(item)
                if bool(doc._.numerize()): # the dictionary is not empty, so there is a number in natural language present in stpwrds
                    remove.append(item)
                    print(item, "successfuly added to removal list!")

        cls = spacy.util.get_lang_class('en')
        # Add
        for word in add:
            cls.Defaults.stop_words.add(word)
            print("Stop word [", word, "] successfully added!")
        # Remove
        for word in remove:
            try:
                cls.Defaults.stop_words.remove(word)
                print("Stop word [", word, "] successfully removed!")  
            except:
                print("Stop word [", word, "] could not be removed because it is not contained in the current set.") 

    stpwrds = sorted(STOP_WORDS)
    if verbose:
        print("\nSTOPWORDS:",len(stpwrds),"\n")
        for word in stpwrds:
            print(word)
    return stpwrds

# Function to lemmatize and remove stop words
def lmtz_and_rmv_stpwrds(text, model='en_core_web_lg', verbose=False):
    """
    Remove stop words and lemmatize text. 
    :param text: Text input from which stop words are removed and which is lemmatized.
    :param model: Pretrained spacy pipeline to use.
    :param verbose: If True, removed stop words are printed.
    :return: Processed text.
    """
    nlp = spacy.load(model)
    doc = nlp(text)
    stpwrds = set(nlp.Defaults.stop_words)
    
    lemmatized_tokens = [token.lemma_ for token in doc if token.lemma_ not in stpwrds]
    
    if verbose:
        sentences = [sent.text for sent in doc.sents]
        lemmatized_sentences = [' '.join([token.lemma_ for token in nlp(sentence) if token.lemma_ not in stpwrds]) for sentence in sentences]
        
        for sentence, lemma_sentence in zip(sentences, lemmatized_sentences):
            highlighted_text = ''
            
            for token in nlp(sentence):
                if token.text.isalpha():  # Exclude non-alphabetic tokens
                    if token.lemma_ not in lemmatized_tokens:
                        highlighted_text += colored(token.text, 'red') + ' '
                    else:
                        highlighted_text += token.text + ' '
                else:
                    highlighted_text += token.text + ' '
            
            print(highlighted_text.strip())
            print(lemma_sentence)
            print('')
    
    return ' '.join(lemmatized_tokens)

