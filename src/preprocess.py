import os
import re
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
    Removes and replaces literals in the text. 

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

def chng_stpwrds(nlp, add=[],remove=[],remove_numbers=False,restore_default=False, default_filename = 'stopwords_en.pickle',verbose=False):
    """
    Adds and removes stop words to the default of SpaCy. 

    :param nlp: Pre-loaded SpaCy model.
    :param add: Stop words to add, e.g. ["by", "the"].
    :param remove: Stop words to remove, e.g. ["to", "one"].
    :param remove_numbers: Flag to remove all numbers in natural language from the stop words list.
    :param restore_default: Flag to restore the default stop word selection provided by SpaCy. 
    :param default_filename: Name of the file with the default stop words. Only relevant if restore_default == True.
    :param verbose: If True, stop words are printed.
    :return: Stop words.
    """

    if restore_default:
        default_filepath = os.path.join('..', 'src', default_filename) # works in src and notebooks folder
        with open(default_filepath, 'rb') as file:
            cls = pickle.load(file)
    else:
        if remove_numbers:
            stpwrds = sorted(STOP_WORDS)

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

def lmtz_and_rmv_stpwrds(nlp ,text, verbose=False):
    """
    Remove stop words and lemmatize text. 

    :param nlp: Pre-loaded SpaCy model.
    :param text: Text input from which stop words are removed and which is lemmatized.
    :param verbose: If True, removed stop words are printed.
    :return: Processed text.
    """
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

def determine_enum_type(sentence, enum_patterns):
    """
    Determines the type of enumeration in the sentence.

    :param sentence: The sentence to check.
    :param enum_patterns: Dictionary of regular expressions serving as enumeration patterns.
    :return: Name of the enumeration pattern if found, None otherwise.
    """
    for pattern_name, pattern in enum_patterns.items():
        # Construct regex for current pattern
        combined_pattern = r"(: NEWLINE " + pattern + r")"

        # Check if the pattern matches
        if re.search(combined_pattern, sentence, re.IGNORECASE):
            return pattern_name

    return None

def get_enum_details(sentence_doc, enum_pattern):
    """
    Extracts detailed information about enumeration items in a sentence.

    :param sentence_doc: SpaCy Doc object of the sentence.
    :param enum_pattern: Regular expression pattern for enumeration items.
    :return: List of tuples with detailed information about each enumeration item.
    """
    enumeration_details = []
    total_enum_item_counter = 0
    level_enum_item_counter = {}
    current_level = 0

    for token in sentence_doc:
        if re.match(enum_pattern, token.text):
            total_enum_item_counter += 1
            current_level += 1
            level_enum_item_counter[current_level] = level_enum_item_counter.get(current_level, 0) + 1
            start_token_number_enumerator = token.i

            # Find the end of the enumeration item
            end_token_number_enum_item = start_token_number_enumerator
            for next_token in sentence_doc[start_token_number_enumerator + 1:]:
                if re.match(enum_pattern, next_token.text):
                    break
                end_token_number_enum_item = next_token.i

            enumeration_details.append((
                enum_pattern,
                current_level,
                total_enum_item_counter,
                level_enum_item_counter[current_level],
                start_token_number_enumerator,
                token.i,
                end_token_number_enum_item
            ))

    return enumeration_details


def split_to_chunks(nlp, text, enum_patterns, separators=['.','!'], exceptions=True, case=None):
    """
    Splits the input into an array of text chunks, which might be a sentence or a sequence of enuemration items.

    :param nlp: Pre-loaded SpaCy model.
    :param text: Input string.
    :param enum_patterns: Dictionary of regular expressions serving as enumeration patterns.
    :param separators: Array where each token determines the separation of the sentences.
    :param exceptions: Determines if exceptions should be considered or not.
    :param case: The use case, for quality control.
    :return: Array of sentences.
    """
    # Replace and handle line breaks and spaces
    text = text.replace("\n\n", "\n").replace("\n \n", "\n")
    text = text.replace("\n", " NEWLINE ").replace("   ", " ").replace("  ", " ")

    doc = nlp(text)

    sentences = []
    enumerations = []
    start = 0

    for token in doc:
        if token.text in separators:
            end = token.idx
            sentence = text[start:end].strip()

            # Check for exceptions if needed
            if exceptions and sentence:
                next_token = token.nbor() if token.i + 1 < len(doc) else None
                prev_token = doc[token.i - 1] if token.i - 1 >= 0 else None
                prev_prev_token = doc[prev_token.i - 1] if prev_token.i - 1 >= 0 else None

                # Define the exception conditions to prevent splits due to:
                is_exception = (
                    # a number, for example: "1.5kg";
                    (prev_token and prev_token.like_num and next_token and next_token.like_num)
                    # "e.g.";
                    or (prev_token and prev_token.text.lower() == "e" and next_token and next_token.text.lower() == "g") 
                    # "eg."" (wrong spelling);
                    or (prev_token and prev_token.text.lower() == "eg") 
                    # multiple sequential separators, for example: "...";
                    or (len(sentence) > 0 and sentence[-1] in separators) 
                    # an enumeration item (assuming it starts at a new line), for example "\n1.";
                    or (prev_prev_token and prev_prev_token.text == "NEWLINE") 
                    # "i.e.";
                    or (prev_token and prev_token.text.lower() == "i" and next_token and next_token.text.lower() == "e") 
                    # special methodology name for CDM use case.
                    or (prev_token and prev_token.text.lower() == "iii" and next_token and next_token.text.lower() == "b")
                )

                if not is_exception:
                    enum_type = determine_enum_type(sentence, enum_patterns)

                    if enum_type is not None:
                        # print("Enumeration found in:", case)
                        # print("Enum type:", enum_type)
                        # Enumeration case
                        end_of_enum = start
                        in_enumeration = True
                        while in_enumeration:
                            next_period = text.find(".", end_of_enum + 1)
                            next_newline = text.find(" NEWLINE ", next_period)
                            if next_period == -1 or next_newline == -1:
                                end_of_enum = len(text)
                                break
                            
                            next_part = text[next_newline:].lstrip(" NEWLINE ")
                            if not re.match(enum_patterns[enum_type], next_part):
                                end_of_enum = next_newline
                                in_enumeration = False
                            else:
                                end_of_enum = next_newline

                        sentence = text[start:end_of_enum]
                        start = end_of_enum + len(" NEWLINE ")
                        enumerations.append()
                    else:
                        start = end + 1
                        enumerations.append()
                    sentence = sentence.replace("NEWLINE", "\n").replace("\n \n", "\n").strip()
                    sentences.append(sentence)

    # Add the last sentence if there is any text left
    if start < len(text):
        last_sentence = text[start:].replace("NEWLINE", "\n").replace("\n \n", "\n").strip()
        if last_sentence:
            sentences.append(last_sentence)

    return sentences, enumerations