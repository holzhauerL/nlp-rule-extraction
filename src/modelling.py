from spacy.matcher import Matcher
from termcolor import colored
from tabulate import tabulate
from collections import defaultdict

class ConstraintSearcher:
    """
    Base class for searching specific constraint patterns in text using spaCy's Matcher. 
    Provides a framework to define and search for patterns related to constraints like inequalities or equalities.

    :param nlp: An instance of a spaCy Language model used for processing text.
    :param parameters: A dictionary containing various settings and parameters for constraint searching.
    """
    def __init__(self, nlp, parameters):
        """
        Initializes the ConstraintSearcher.

        :param nlp: An instance of a spaCy Language model used for processing text.
        :param parameters: A dictionary containing various settings and parameters for constraint searching.
        """
        self.nlp = nlp
        self.parameters = parameters
        self.constraints = {'id': [], 'type': [], 'match': [], 'patterns': [], 'exception': [], 'negation': [], 'symbol': [],  'level': [], 'predecessor': [], 'successor': []}
        self.type = None

    @staticmethod
    def no_digit_or_period():
        """
        Generates a spaCy pattern that matches tokens which are not digits or periods. 
        This pattern is used to capture textual elements in a constraint expression.

        :return: A list of dictionaries representing the spaCy pattern for matching non-digit and non-period tokens.
        """
        return [{"TEXT": {"NOT_IN": ["."]}, "IS_DIGIT": False, "OP": "*"}]

    def expand_dict(self, input_dict):
        """
        Expands the given dictionary by adding modified key-value pairs. 
        This is specifically used to handle variations in constraint expressions like "greater than" and "greater than or equal to".

        :param input_dict: A dictionary with keys as phrases and values as their corresponding symbols.
        :return: An expanded dictionary containing additional key-value pairs for expanded expressions.
        """
        expanded_dict = {}
        for phrase, symbol in input_dict.items():
            expanded_dict[phrase] = symbol
            if symbol in {"<", ">"}:
                modified_phrase_or_equal_to = phrase + " or equal to"
                modified_symbol_or_equal_to = symbol + "="
                expanded_dict[modified_phrase_or_equal_to] = modified_symbol_or_equal_to
                if "than" in phrase:
                    modified_phrase_than_replaced = phrase.replace("than", "or equal to")
                    expanded_dict[modified_phrase_than_replaced] = modified_symbol_or_equal_to
        return expanded_dict

    def check_for_negation(self, doc, start_index, negation_tokens, window_size):
        """
        Checks for negation tokens within a specified window around a start index in a spaCy Doc object.

        :param doc: A spaCy Doc object representing the processed text.
        :param start_index: The index of the token in the Doc from where to start checking for negation.
        :param negation_tokens: A list of tokens considered as negations.
        :param window_size: A tuple indicating the window size (pre-index, post-index) to check for negation.
        :return: Token which triggered the negation and boolean indicating whether a negation token is found within the specified window.
        """
        for i in range(max(0, start_index - window_size[0]), min(len(doc), start_index + window_size[1])):
            if doc[i].lower_ in negation_tokens:
                return doc[i].i, True
        return None, False

    def highlight_matches(self, text, matches, negations, types):
        """
        Highlights the matches in the text based on their types and negations. 
        Uses different colors for different types of matches and negations. 
        Ensures that a token is highlighted only once.

        :param text: The original text in which matches are to be highlighted.
        :param matches: A list of match tuples (start_index, end_index) or ((start_index, end_index), token_index) for 'META' type.
        :param negations: A list of indices where negations occur.
        :param type: A list of types corresponding to each match.
        :return: None. The function prints the highlighted text.
        """
        doc = self.nlp(text)
        highlighted_text = ""
        last_index = 0
        combined = []
        highlighted_indices = set()  # Set to keep track of highlighted tokens

        # Remove duplicates
        negations = list(set(negations))

        for match, mtype in zip(matches, types):
            if mtype != 'META':
                combined.append((match[0], match[1], 'match', mtype))
            else:
                # Handle the structure of the META match
                if isinstance(match[0], tuple):  # ((start, end_enum), end)
                    range_match, single_token = match
                    combined.append((range_match[0], range_match[1], 'match', mtype))
                else:  # (start, end)
                    combined.append((match[0], match[1], 'match', mtype))

        combined.extend([(negation, negation + 1, 'negation', None) for negation in negations])
        combined.sort(key=lambda x: x[0])

        for start, end, type, mtype in combined:
            if start in highlighted_indices:
                continue  # Skip if start index is already highlighted

            start_char = doc[start].idx
            end_char = doc[end - 1].idx + len(doc[end - 1].text)
            highlighted_text += text[last_index:start_char]

            # Determine color based on type and match type
            color = 'magenta'  # Default color for other types
            if type == 'match':
                if mtype == 'INEQ':
                    color = 'green'
                elif mtype == 'EQ':
                    color = 'blue'
                elif mtype == 'META':
                    color = 'yellow'
            elif type == 'negation':
                color = 'red'

            highlighted_text += colored(text[start_char:end_char], color, attrs=['bold'])
            last_index = end_char

            # Update highlighted indices
            for i in range(start, end):
                highlighted_indices.add(i)

        highlighted_text += text[last_index:]
        print(highlighted_text)

    def search_matches(self, text, id):
        """
        Searches for matches in the text based on defined patterns, considering exceptions.

        :param text: The text in which to search for matches.
        :param id: The ID of the first match found by the current function within the document.
        :return id: The ID of the last match + 1 found by the current function within the document.
        """
        # Merging general and exception patterns
        general_dict = self.expand_dict(self.parameters["general"])
        exceptions_dict = {key: value for key, value in self.parameters["exceptions"].items()}
        merged_dict = {**general_dict, **exceptions_dict}

        # Creating patterns
        patterns = self.create_patterns(merged_dict)
        formatted_exception_patterns = {key.upper().replace(" ", "_"): value for key, value in self.parameters["exception_patterns"].items()}
        for key, value in formatted_exception_patterns.items():
            if callable(value):
                patterns[key] = value()
            else:
                patterns[key] = value

        # Matching patterns
        matcher = Matcher(self.nlp.vocab)
        for key, pattern in patterns.items():
            matcher.add(key, [pattern])
        doc = self.nlp(text)
        matches = matcher(doc)

        # Processing matches
        seen_tokens = set()
        for match_id, start, end in matches:
            if end not in seen_tokens:
                match_str = self.nlp.vocab.strings[match_id]
                formatted_match_str = match_str.upper().replace(" ", "_")
                negated_token, is_negated = self.check_for_negation(doc, start, self.parameters["negation_tokens"], self.parameters["window_size"])

                # Determining if the match is an exception
                # Check in formatted exception patterns and the keys of exceptions_dict after formatting them
                is_exception = formatted_match_str in formatted_exception_patterns or \
                            formatted_match_str in {key.upper().replace(" ", "_") for key in exceptions_dict.keys()}

                self.constraints['id'].append(id)
                self.constraints['type'].append(self.type)
                self.constraints['match'].append((start, end))
                self.constraints['patterns'].append(formatted_match_str)
                self.constraints['exception'].append(is_exception)
                self.constraints['negation'].append((negated_token, is_negated))
                self.constraints['level'].append('') # Only relevant for enumeration constraints

                connector_pre = 'START' if id == 1 else 'FOLLOW' # Indicate the start of the constraints
                self.constraints['predecessor'].append((id-1,connector_pre))
                self.constraints['successor'].append((id+1,'FOLLOW'))

                # Updating the ID
                id += 1

                # Handling symbols
                phrase = match_str.replace("_", " ").lower()
                symbol = merged_dict.get(phrase, None)
                if symbol is not None and is_negated:
                    symbol = self.parameters["negation_operators"].get(symbol, symbol)
                self.constraints['symbol'].append(symbol)
                seen_tokens.add(end)

        return id

    def create_patterns(self, expanded_dict):
        """
        Abstract method to be implemented by subclasses. It creates specific patterns for searching constraints based on an expanded dictionary.

        :param expanded_dict: A dictionary with expanded phrases and corresponding symbols.
        :return: A dictionary of patterns to be used by the spaCy Matcher.
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    
class InequalityConstraintSearcher(ConstraintSearcher):
    """
    A subclass of ConstraintSearcher specifically designed for searching inequality constraints in text.
    It handles expressions related to inequalities such as 'greater than', 'less than', etc.
    """
    def __init__(self, nlp, parameters):
        """
        Initializes the InequalityConstraintSearcher with the given NLP model and parameters.

        :param nlp: An instance of a spaCy Language model used for processing text.
        :param parameters: A dictionary containing settings and parameters for inequality constraint searching.
        """
        super().__init__(nlp, parameters)
        self.type = 'INEQ'

    def create_patterns(self, expanded_dict):
        """
        Creates patterns for inequality expressions based on the expanded dictionary.

        This method generates spaCy patterns that are used to identify inequality expressions in text, like 'greater than 10', 'less than or equal to 5', etc.

        :param expanded_dict: A dictionary with expanded phrases and corresponding symbols for inequalities.
        :return: A dictionary of spaCy patterns for identifying inequality constraints.
        """
        patterns = {}
        for phrase in expanded_dict.keys():
            phrase_pattern = [{"LOWER": word} for word in phrase.split()]
            phrase_pattern.extend(ConstraintSearcher.no_digit_or_period())
            phrase_pattern.append({"LIKE_NUM": True})
            patterns[phrase] = phrase_pattern
        return patterns

class EqualityConstraintSearcher(ConstraintSearcher):
    """
    A subclass of ConstraintSearcher specifically designed for searching equality constraints in text.
    This class is focused on expressions that denote equality, such as 'equal to', 'same as', etc.
    """
    def __init__(self, nlp, parameters):
        """
        Initializes the EqualityConstraintSearcher with the given NLP model and parameters.

        :param nlp: An instance of a spaCy Language model used for processing text.
        :param parameters: A dictionary containing settings and parameters for equality constraint searching.
        """
        super().__init__(nlp, parameters)
        self.type = 'EQ'

    def create_patterns(self, expanded_dict):
        """
        Creates patterns for equality expressions based on the expanded dictionary.

        This method generates spaCy patterns that are used to identify equality expressions in text, triggered by keywords like 'must' or 'shall'.

        :param expanded_dict: A dictionary with expanded phrases and corresponding symbols for equalities.
        :return: A dictionary of spaCy patterns for identifying equality constraints.
        """
        patterns = {}
        for phrase in expanded_dict.keys():
            phrase_pattern = [{"LOWER": word} for word in phrase.split()]
            patterns[phrase.upper().replace(" ", "_")] = phrase_pattern
        return patterns
    
class MetaConstraintSearcher(ConstraintSearcher):
    """
    A subclass of ConstraintSearcher specifically designed for searching meta constraints in text.
    Meta constraints encapsulate one or multiple constraints.
    """
    def __init__(self, nlp, parameters):
        """
        Initializes the MetaConstraintSearcher with the given NLP model and parameters.

        :param nlp: An instance of a spaCy Language model used for processing text.
        :param parameters: A dictionary containing settings and parameters for meta constraint searching.
        """
        super().__init__(nlp, parameters)
        self.type = 'META'

    def search_enum_constraints(self, text, enumeration_summary, id):
        """
        Searches and processes meta constraints based on enumeration summaries.

        :param text: The text in which to search for matches.
        :param enumeration_summary: A tuple containing enumeration details from the dataframe.
        :param id: The ID of the first meta constraint found by the current function within the document.
        :return: The updated ID.
        """
        logical_connections = {}
        exceptions = {}
        pattern_names = {}
        last_level = 0
        levels_covered = []

        doc = self.nlp(text)

        # Create the patterns for the exceptions
        patterns = {}
        for phrase in self.parameters['enum_exceptions'].keys():
            phrase_pattern = [{"LOWER": word} for word in phrase.split()]
            patterns[phrase.upper().replace(" ", "_")] = phrase_pattern

        for enum in enumeration_summary:
            level, start_token_enumeration, end_token_enumeration, end_token_item = enum[1], enum[-3], enum[-2], enum[-1]

            if last_level != level and level not in levels_covered:
                first_enum_item = True
                logical_connections[level] = 'AND'
                exceptions[level] = False
                pattern_names[level] = 'ENUM'
                levels_covered.append(level)

            # Determine the logical connection for the enumeration (AND/OR), only for the first enumeration item
            if first_enum_item:
                # Matching patterns only in the text preceding the first enumeration item
                matcher = Matcher(self.nlp.vocab)
                for key, pattern in patterns.items():
                    matcher.add(key, [pattern])
                matches = matcher(doc[:start_token_enumeration])

                if doc[end_token_item].lower_ == 'or':
                    logical_connections[level] = 'OR'
                    exceptions[level] = True
                    pattern_names[level] = 'ENUM_OR'
                elif len(matches):
                    match = matches[-1] # only considering the last match
                    match_str = self.nlp.vocab.strings[match[0]]
                    formatted_match_str = match_str.lower().replace("_"," ")
                    logical_connections[level] = self.parameters['enum_exceptions'][formatted_match_str]
                    exceptions[level] = True
                    pattern_names[level] = 'ENUM_' + match_str

                first_enum_item = False

            # Filling in the details for the meta constraint
            self.constraints['id'].append(id)
            self.constraints['type'].append(self.type)
            self.constraints['match'].append(((start_token_enumeration, end_token_enumeration),end_token_item))
            self.constraints['patterns'].append(pattern_names[level])
            self.constraints['exception'].append(exceptions[level])
            self.constraints['negation'].append('')
            self.constraints['symbol'].append('')
            self.constraints['level'].append(level)

            # Set predecessor and successor
            connector_pre = 'START' if id == 1 else logical_connections[level]
            connector_suc = logical_connections[level]
            self.constraints['predecessor'].append((id-1, connector_pre))
            self.constraints['successor'].append((id+1, connector_suc))

            id += 1

            last_level = level

        return id

    def search_if_clauses(self, text, succeeding_linebreaks, id):
        """
        Searches for 'if' clauses within a single line of text and records their details.

        This method uses spaCy pattern matching to identify 'if-then' or 'if-comma' structures within the text. 
        If multiple matches start with the same 'if' token, the longest match is selected. The search is limited 
        to single lines, considering linebreaks in the text.

        The end of a match is extended to the last comma or 'then', whichever is later in the text.

        :param text: The text in which to search for 'if' clauses.
        :param succeeding_linebreaks: A list indicating linebreaks in the text.
        :param id: The ID to start with for the first 'if' clause found.
        :return: The updated ID.
        """
        doc = self.nlp(text)
        matcher = Matcher(self.nlp.vocab)
        for pattern_name, pattern_func in self.parameters['if_patterns'].items():
            matcher.add(pattern_name, [pattern_func()])

        matches = matcher(doc)
        processed_matches = {}

        for match_id, start, end in matches:
            if not succeeding_linebreaks[start:end].count(True):  # Ensure match is within a single line
                # Extend the match end to the last comma or 'then'
                extended_end = end
                for i in range(end, len(doc)):
                    if doc[i].lower_ == ',' or doc[i].lower_ == 'then':
                        extended_end = i + 1
                    if succeeding_linebreaks[i]:
                        break  # Stop at line break

                if start not in processed_matches or (extended_end - start) > (processed_matches[start][1] - processed_matches[start][0]):
                    processed_matches[start] = (start, extended_end)

        for start, (start, end) in processed_matches.items():
            self.constraints['id'].append(id)
            self.constraints['type'].append(self.type)
            self.constraints['match'].append((start, end))
            self.constraints['patterns'].append(self.nlp.vocab.strings[match_id].upper())
            self.constraints['exception'].append(False)
            self.constraints['negation'].append('')
            self.constraints['symbol'].append('')
            self.constraints['level'].append('')

            connector_pre = 'START' if id == 1 else 'AND'
            self.constraints['predecessor'].append((id-1, connector_pre))
            self.constraints['successor'].append((id+1, 'AND'))

            id += 1

        return id

    def search_for_clauses(self):
        # Placeholder for 'search for clauses'
        pass

    def search_connectors(self):
        # Placeholder for 'search connectors'
        pass

    def rank_constraints(self):
        # Placeholder for 'rank constraints'
        pass

def search_constraints(nlp, text, equality_params, inequality_params, meta_params, enumeration_summary, linebreaks, id=1):
    """
    Conducts a combined search for equality, inequality, and meta (enumeration) constraints within a given text.

    This function initializes InequalityConstraintSearcher, EqualityConstraintSearcher, and MetaConstraintSearcher, performs the search, and then combines their findings.

    :param nlp: A spaCy Language model instance used for text processing.
    :param text: The text to search within for constraints.
    :param equality_params: Parameters for the EqualityConstraintSearcher.
    :param inequality_params: Parameters for the InequalityConstraintSearcher.
    :param meta_params: Parameters for the MetaConstraintSearcher.
    :param enumeration_summary: Enumeration summaries for meta constraint searching.
    :param linebreaks: Information on succeding line breaks for meta constraint searching.
    :param id: The ID of the first match found by the current function within the document.
    :return constraints: A dictionary with detailed information about the found constraints. 
    :return id: The ID of the last match + 1 found by the current function within the document.
    """
    # Initialize searchers
    inequality = InequalityConstraintSearcher(nlp, inequality_params)
    equality = EqualityConstraintSearcher(nlp, equality_params)
    meta = MetaConstraintSearcher(nlp, meta_params)

    # Perform the searches and combine match details
    id = inequality.search_matches(text, id)
    id = equality.search_matches(text, id)
    id = meta.search_enum_constraints(text, enumeration_summary, id)
    id = meta.search_if_clauses(text, linebreaks, id)

    # Combine matches, types, negations for visualisation
    combined_negations = [neg[0] for neg in inequality.constraints['negation'] if neg[1]] + \
                         [neg[0] for neg in equality.constraints['negation'] if neg[1]] + \
                         [] * (len(meta.constraints['match']))

    # Create output dict
    constraints = {}
    for key in inequality.constraints.keys():
        constraints[key] = inequality.constraints[key] + equality.constraints[key] + meta.constraints[key]

    # Highlighting text
    ConstraintSearcher(nlp, equality_params).highlight_matches(text, constraints['match'], combined_negations, constraints['type'])

    # Print in tabular format
    combined_table_data = zip(*constraints.values()) 
    # Only print if any of the lists in constraints has elements
    if any(constraints.values()): 
        print()
        print(tabulate(combined_table_data, headers=constraints.keys()))

    return constraints, id

def search_constraints_in_data(nlp, data, equality_params, inequality_params, meta_params):
    """
    Conducts a combined search for inequality, equality and meta constraints for all use cases.

    Wrapper function for search_constraints(). 

    :param nlp: A spaCy Language model instance used for text processing.
    :param data: A dictionary with one dataframe per use case containing pre-processed chunks of text in the 'Lemma' column.  
    :param equality_params: Parameters for the EqualityConstraintSearcher.
    :param inequality_params: Parameters for the InequalityConstraintSearcher.
    :param meta_params: Parameters for the MetaConstraintSearcher.
    :return constraints: A dictionary with detailed information about the found constraints. 
    """
    delimiter_line = "+"*80
    constraints_tmp = defaultdict(lambda: defaultdict(list)) # Enable dict assignment before knowing the keys

    for use_case, df in data.items():
        print()
        print(delimiter_line)
        print(delimiter_line, "\n")
        print(use_case.upper(), "\n")
        print(delimiter_line)
        print(delimiter_line, "\n")

        id = 1  # Initialize ID for each use case
        for index, row in df.iterrows(): # Iterate over the rows in the dataframe
            for chunk_index, chunk in enumerate(row['Lemma']):  # Iterate over each chunk in the Lemma column
                print()
                print("++++ CHUNK ++++", "\n")
                new_constraints, id = search_constraints(nlp, chunk, equality_params, inequality_params, meta_params, row['Enumeration'][chunk_index], row['Linebreak'][chunk_index], id)
                for key, values in new_constraints.items():
                    constraints_tmp[use_case][key].extend(values)
                    if key == 'ID':  # Only append to 'Index' and 'Chunk' when new 'ID' is found
                        constraints_tmp[use_case]['Index'].extend([index] * len(values))
                        constraints_tmp[use_case]['Chunk'].extend([chunk_index] * len(values))

        # Overwrite the successor of the last found constraint item to demark the end, if any constraint was found
        if constraints_tmp[use_case]['Successor']:
            constraints_tmp[use_case]['Successor'][-1] = (constraints_tmp[use_case]['Successor'][-1][0],'END')

    # Format to regular dict
    constraints = dict()
    for key, value in constraints_tmp.items():
        constraints[key] = dict(value)

    return constraints