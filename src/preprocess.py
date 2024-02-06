import re
import numerizer
from termcolor import colored
import pandas as pd
from spacy.matcher import Matcher
from spacy.lang.en.stop_words import STOP_WORDS


class Preprocessor:
    """
    Class for reading .txt files, preprocessing the text, and splitting into sentences, while also extracting enumeration details.
    """
    def __init__(self, nlp, parameters, linebreak, input_file_paths, verbose):
        """
        Initializes the Preprocessor.

        :param nlp: An instance of a spaCy Language model used for processing text.
        :param parameters: A dictionary containing various settings and parameters for preprocessing.
        :param linebreak: The character replacing a linebreak to split into chunks.
        :param file_paths: A dictionary with the paths to the .txt input files.
        :param verbose: Parameter to control the printed output. If True, output is printed.
        """
        self.nlp = nlp
        self.parameters = parameters
        self.linebreak = linebreak
        self.file_paths = input_file_paths
        self.verbose = verbose
        self.stop_word_changes = {}

    def txt_to_df(self, file_path, section_start="# ", subsection_start="## "):
        """
        Reads the text from a .txt file in a certain format, breaks it down into lines and returns it as a pandas Dataframe, organised into sections and subsections. 

        :param file_path: Relative path to the .txt input file.
        :param section_start: String indicating the start of a section title.
        :param subsection_start: String indicating the start of a subsection title.
        :return: Dataframe with the columns "Section", "Raw", "Chunks, "Lemma", "Alignment", "Linebreak" and "Enumeration" as well as {current_section}/{current_subsection} as a syntax for concatinating sections and sub-sections. 
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

        data = {"Section": [], "Raw": [], "Chunks": [], "Lemma": [], "Alignment": [],"Linebreak": [], "Enumeration": []}
        
        for column_name, column_content in columns:
            data["Section"].append(column_name)
            data["Raw"].append('\n'.join(column_content))
            data["Chunks"].append('')
            data["Lemma"].append('')
            data["Alignment"].append('')
            data["Linebreak"].append('')
            data["Enumeration"].append('')

        return pd.DataFrame(data)

    def replace_and_remove(self, text, replace, remove, space=True):
        """
        Replaces and removes literals in the text. 

        :param text: Input string.
        :param replace: Mapping of strings to replace.
        :param remove: List of strings to remove.
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
    
    def change_stop_words(self, add=[],remove=[],remove_numbers=False,restore_default=False):
        """
        Adds and removes stop words to/from the default set in SpaCy, with options to save/restore customizations.

        :param add: Stop words to add, e.g. ["by", "the"].
        :param remove: Stop words to remove, e.g. ["to", "one"].
        :param remove_numbers: Flag to remove all numbers in natural language from the stop words list.
        :param restore_default: Flag to restore the default stop word selection provided by SpaCy.
        """
        if not restore_default:
            if remove_numbers:
                stpwrds = sorted(STOP_WORDS)

                for item in stpwrds:
                    doc = self.nlp(item)
                    if bool(doc._.numerize()): # the dictionary is not empty, so there is a number in natural language present in stpwrds
                        remove.append(item)
                        if self.verbose:
                            print(item, "successfuly added to removal list!")
            
            self.stop_word_changes = {'add': add, 'remove': remove}

        else:
            # Restore default by inverting saved additions and removals
            add, remove = self.stop_word_changes['remove'], self.stop_word_changes['add']
            if self.verbose:
                print("Restored default stop words and deleted the backup file.")
            else:
                if self.verbose:
                    print("No saved stop words changes to restore.")

        # Add
        for word in add:
            self.nlp.Defaults.stop_words.add(word)
            if self.verbose: 
                print("Stop word [", word, "] successfully added!")
        # Remove
        for word in remove:
            try:
                self.nlp.Defaults.stop_words.remove(word)
                if self.verbose: 
                    print("Stop word [", word, "] successfully removed!")  
            except:
                if self.verbose: 
                    print("Stop word [", word, "] could not be removed because it is not contained in the current set.") 

        stpwrds = sorted(self.nlp.Defaults.stop_words)
        if self.verbose:
            print("\nSTOPWORDS:",len(stpwrds),"\n")
            for word in stpwrds:
                print(word)

    def lemmatize_and_remove_stop_words(self, text):
        """
        Remove stop words and lemmatize text. This function also tracks if the token was originally followed by a line break.
        Parts of the code taken from https://github.com/explosion/spaCy/issues/7735. 

        :param text: Text input from which stop words are removed and which is lemmatized.
        :return: Processed text, alignment mapping (output to input), and line break status.
        """
        stpwrds = set(self.nlp.Defaults.stop_words)
        alignment = {}  # Mapping output to input
        output_tokens = []
        suc_linebreaks = []  # List to keep track of line breaks
        output_length = 0  # Cumulative length of the output string
        input_index_offset = 0  # Offset to track index in original text

        lines = text.split('\n')
        for line in lines:
            doc = self.nlp(line)
            for tok in doc:
                if tok.is_space or tok.lemma_ in stpwrds:
                    continue

                output_tokens.append(tok.lemma_)  # Add the lemmatized token
                # Calculate the start index of this token in the output string
                start_idx_output = output_length
                # Update the output length (including a space for separation)
                output_length += len(tok.lemma_) + 1
                # Map the start index in the output to the start index in the input
                alignment[start_idx_output] = tok.idx + input_index_offset

                suc_linebreaks.append(False)

            # Update the input index offset for the next line
            input_index_offset += len(line) + 1

            # Add a line break marker at the end of each line except the last
            if line != lines[-1]:
                suc_linebreaks[-1] = True

        # Remove redundant commas from output_tokens
        processed_output_tokens = []
        prev_token = None
        for token in output_tokens:
            if token == "," and prev_token == ",":
                continue  # Skip this token if it's a redundant comma
            processed_output_tokens.append(token)
            prev_token = token
        output_tokens = processed_output_tokens
        
        if self.verbose:
            sentences = [sent.text for sent in doc.sents]
            lemmatized_sentences = [' '.join([token.lemma_ for token in self.nlp(sentence) if token.lemma_ not in stpwrds]) for sentence in sentences]
            
            for sentence, lemma_sentence in zip(sentences, lemmatized_sentences):
                highlighted_text = ''
                
                for token in self.nlp(sentence):
                    if token.text.isalpha():  # Exclude non-alphabetic tokens
                        if token.lemma_ not in output_tokens:
                            highlighted_text += colored(token.text, 'red') + ' '
                        else:
                            highlighted_text += token.text + ' '
                    else:
                        highlighted_text += token.text + ' '
                
                print(highlighted_text.strip())
                print(lemma_sentence)
                print('')
        
        output = ' '.join(output_tokens)
        return output, alignment, suc_linebreaks

    def determine_enum_type(self, sentence, enum_patterns, linebreak):
        """
        Determines the type of enumeration in the sentence.

        :param sentence: The sentence to check.
        :param enum_patterns: Dictionary of regular expressions serving as enumeration patterns.
        :param linebreak: The string replacing a line break.
        :return: Name of the enumeration pattern if found, None otherwise.
        """
        for pattern_name, pattern in enum_patterns.items():
            # Construct regex for current pattern
            combined_pattern = rf"(:{linebreak}" + pattern + r")"

            # Check if the pattern matches
            if re.search(combined_pattern, sentence, re.IGNORECASE):
                return pattern_name

        return None

    def split_to_chunks(self, text, enum_patterns, linebreak, separators=['.','!'], exceptions=True):
        """
        Splits the input into an array of text chunks, which might be a sentence or a sequence of enuemration items.

        :param text: Input string.
        :param enum_patterns: Dictionary of regular expressions serving as enumeration patterns.
        :param linebreak: The string replacing a line break to split into chunks.
        :param separators: Array where each token determines the separation of the sentences.
        :param exceptions: Determines if exceptions should be considered or not.
        :return: Array of sentences.
        """
        doc = self.nlp(text)

        sentences = []
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
                        or (prev_prev_token and prev_prev_token.text == linebreak.strip()) 
                        # "i.e.";
                        or (prev_token and prev_token.text.lower() == "i" and next_token and next_token.text.lower() == "e") 
                        # special methodology name for CDM use case.
                        or (prev_token and prev_token.text.lower() == "iii" and next_token and next_token.text.lower() == "b")
                    )

                    if not is_exception:
                        enum_type = self.determine_enum_type(sentence, enum_patterns, linebreak)

                        if enum_type is not None:
                            end_of_enum = start
                            in_enumeration = True
                            while in_enumeration:
                                next_period = text.find(".", end_of_enum + 1)
                                next_newline = text.find(linebreak, next_period)
                                if next_period == -1 or next_newline == -1:
                                    end_of_enum = len(text)
                                    break
                                
                                next_part = text[next_newline:].lstrip(linebreak)
                                if not re.match(enum_patterns[enum_type], next_part):
                                    end_of_enum = next_newline
                                    in_enumeration = False
                                else:
                                    end_of_enum = next_newline

                            sentence = text[start:end_of_enum]
                            start = end_of_enum + len(linebreak)
                        else:
                            start = end + 1
                        sentence = sentence.replace(linebreak.strip(), "\n").strip()
                        sentences.append(sentence)

        # Add the last sentence if there is any text left
        if start < len(text):
            last_sentence = text[start:].replace(linebreak.strip(), "\n").strip()
            if last_sentence:
                sentences.append(last_sentence)

        return sentences
    
    def get_enum_details(self, lemmatized_chunk, succeding_linebreaks, enum_patterns_spacy):
        """
        Analyze a lemmatized chunk of text to identify and summarize enumeration items using spaCy Matcher. 
        This function tracks the enumeration type, its level, and the positions of enumeration items in the text.

        :param lemmatized_chunk: The lemmatized text chunk to analyze.
        :param succeding_linebreaks: A list of booleans indicating if a line break succeeds each token.
        :param enum_patterns_spacy: Dictionary of spaCy token patterns for enumeration matching.

        :return: A list of tuples. Each tuple contains the following elements for an enumeration item:
            - enum_type (str): The type of enumeration (e.g., bullet, number).
            - current_level (int): The hierarchical level of the enumeration item.
            - total_enum_counter (int): A running total count of enumeration items found.
            - level_enum_counter (int): The count of enumeration items at the current level.
            - start_token_number_enumerator (int): The token index where the enumeration item starts.
            - end_token_number_enumerator (int): The token index where the enumeration item ends.
            - end_token_number_enum_item (int): The token index where the content of the enumeration item ends.
        """
        # Create a matcher with the given vocab
        matcher = Matcher(self.nlp.vocab)

        # Add enumeration patterns to the matcher
        for enum_type, pattern in enum_patterns_spacy.items():
            matcher.add(enum_type, [pattern])

        # Process the lemmatized chunk to create a doc object
        doc = self.nlp(lemmatized_chunk)

        # Find matches using the matcher in the doc
        matches = matcher(doc)

        # Initialize variables for enumeration summary, counters, and stacks
        summary = []
        total_counter, previous_start = 0, None
        level_counters, enum_stack = {}, []

        # Iterate through each match found
        for match_id, start, end in matches:
            # Check if a linebreak succeeds the current token
            if start > 0 and succeding_linebreaks[start - 1]:
                # Get the enumeration type from the match
                enum_type = self.nlp.vocab.strings[match_id]
                total_counter += 1

                # Determine the current level of the enumeration
                current_level = next((i + 1 for i, (etype, _) in enumerate(enum_stack) if etype == enum_type), len(enum_stack) + 1)

                # Add or update the enum type in the stack
                if not enum_stack or enum_stack[-1][0] != enum_type:
                    enum_stack.append((enum_type, current_level))

                # Update the counter for the current level
                level_counters.setdefault(current_level, 0)
                level_counters[current_level] += 1

                # Update the end token number for the previous enumeration item
                if previous_start is not None:
                    summary[-1] = summary[-1][:6] + (start - 1,)

                # Remember the start token number for the current enumeration item
                previous_start = start

                # Adjust for a speciality in a pattern
                try:
                    if enum_type == 'number_with_dot':
                        end -= 1
                except:
                    pass

                # Append the current enumeration item details to the summary
                summary.append((enum_type, current_level, total_counter, level_counters[current_level], start, end - 1, -1))

                # Pop the stack if the current enum type differs from the last one
                while enum_stack and enum_stack[-1][0] != enum_type:
                    enum_stack.pop()

        # Ensure that an enumeration has at least two items, otherwise it is not considered an enumeration
        if len(summary) <= 1:
            summary = []

        # Update the end token number for the last enumeration item
        if summary:
            summary[-1] = summary[-1][:6] + (len(doc) - 1,)

        return summary

    def optimized_preprocessing(self):
        """
        An optimized function that combines reading a dictionary of files, preprocessing the text, and splitting into sentences, while also extracting enumeration details.

        :return: Dictionary with one DataFrame for each file containing the preprocessed texts.
        """
        data = {}

        # Add and remove custom stop words to enable the matching
        self.change_stop_words(add=self.parameters['add_stop_words'],remove=self.parameters['non_stop_words'], remove_numbers=True)

        for case, path in self.file_paths.items():

            df = self.txt_to_df(path)
            
            for index, row in df.iterrows():
                # Removing and replacing special characters
                processed_text = self.replace_and_remove(row['Raw'], self.parameters['replace'], self.parameters['remove'])
                # Splitting into chunks
                chunks = self.split_to_chunks(processed_text, self.parameters['enum_patterns_regex'], self.linebreak)
                df.at[index, 'Chunks'] = chunks
                # Check if chunks is not empty
                if chunks:
                    # Lemmatizing and removing stop words
                    results = [self.lemmatize_and_remove_stop_words(chunk) for chunk in chunks]
                    lemmatized_chunks, alignments, succeding_linebreaks = zip(*results)
                    # Analyzing the enumeration structure
                    enumeration_details = [self.get_enum_details(chunk, linebreaks, self.parameters['enum_patterns_spacy']) for chunk, linebreaks in zip(lemmatized_chunks, succeding_linebreaks)]
                else:
                    # Empty chunks
                    lemmatized_chunks, alignments, succeding_linebreaks, enumeration_details = [], [], [], []

                df.at[index, 'Lemma'] = lemmatized_chunks
                df.at[index, 'Alignment'] = alignments
                df.at[index, 'Linebreak'] = succeding_linebreaks
                df.at[index, 'Enumeration'] = enumeration_details
            
            data[case] = df
        
        # Restore the default set of stop words for the rule-based constraint building
        self.change_stop_words(restore_default=True)

        return data