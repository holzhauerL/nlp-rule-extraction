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
        self.unique_matches = {'id': [], 'type': [], 'match': [], 'patterns': [], 'exception': [], 'negation': [], 'symbol': [],  'predecessor': [], 'successor': []}
        self.match_type = None

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
        :return: Boolean indicating whether a negation token is found within the specified window.
        """
        for i in range(max(0, start_index - window_size[0]), min(len(doc), start_index + window_size[1])):
            if doc[i].lower_ in negation_tokens:
                return True
        return False

    def highlight_matches(self, text, matches, negations, match_type):
        """
        Highlights the matches in the text based on their types and negations. 
        It uses different colors for different types of matches and negations.

        :param text: The original text in which matches are to be highlighted.
        :param matches: A list of match tuples (start_index, end_index).
        :param negations: A list of indices where negations occur.
        :param match_type: A list of types corresponding to each match.
        :return: None. The function prints the highlighted text.
        """
        doc = self.nlp(text)
        highlighted_text = ""
        last_index = 0
        combined = [(start, end, 'match', mtype) for (start, end), mtype in zip(matches, match_type)]
        combined.extend([(negation, negation + 1, 'negation', None) for negation in negations])
        combined.sort(key=lambda x: x[0])
        for start, end, type, mtype in combined:
            start_char = doc[start].idx
            end_char = doc[end - 1].idx + len(doc[end - 1].text)
            highlighted_text += text[last_index:start_char]
            if type == 'match' and mtype == 'INEQ':
                color = 'green' 
            elif type == 'match':
                color = 'blue'
            else:
                color = 'red'
            highlighted_text += colored(text[start_char:end_char], color, attrs=['bold'])
            last_index = end_char
        highlighted_text += text[last_index:]
        print(highlighted_text)

    def search_matches(self, text, exception_patterns, id):
        """
        Searches for matches in the text based on defined patterns, considering exceptions.

        :param text: The text in which to search for matches.
        :param exception_patterns: A dictionary of patterns that should be treated as exceptions.
        :param id: The ID of the first match found by the current function within the document.
        :return unique_matches: A dictionary containing details of the unique matches found in the text.
        :return id: The ID of the last match + 1 found by the current function within the document.
        """
        # Merging general and exception patterns
        general_dict = self.expand_dict(self.parameters["general"])
        exceptions_dict = {key: value for key, value in self.parameters["exceptions"].items()}
        merged_dict = {**general_dict, **exceptions_dict}

        # Creating patterns
        patterns = self.create_patterns(merged_dict)
        formatted_exception_patterns = {key.upper().replace(" ", "_"): value for key, value in exception_patterns.items()}
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
                is_negated = self.check_for_negation(doc, start, self.parameters["negation_tokens"], self.parameters["window_size"])
                negated_token = start - 1 if is_negated else None

                # Determining if the match is an exception
                # Check in formatted exception patterns and the keys of exceptions_dict after formatting them
                is_exception = formatted_match_str in formatted_exception_patterns or \
                            formatted_match_str in {key.upper().replace(" ", "_") for key in exceptions_dict.keys()}

                self.unique_matches['id'].append(id)
                self.unique_matches['type'].append(self.match_type)
                self.unique_matches['match'].append((start, end))
                self.unique_matches['patterns'].append(formatted_match_str)
                self.unique_matches['exception'].append(is_exception)
                self.unique_matches['negation'].append((is_negated, negated_token))

                connector_pre = 'START' if id == 1 else 'FOLLOW' # Indicate the start of the constraints
                self.unique_matches['predecessor'].append((id-1,connector_pre))
                self.unique_matches['successor'].append((id+1,'FOLLOW'))

                # Updating the ID
                id += 1

                # Handling symbols
                phrase = match_str.replace("_", " ").lower()
                symbol = merged_dict.get(phrase, None)
                if symbol is not None and is_negated:
                    symbol = self.parameters["negation_operators"].get(symbol, symbol)
                self.unique_matches['symbol'].append(symbol)
                seen_tokens.add(end)

        return self.unique_matches, id

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
        self.match_type = 'INEQ'

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
        self.match_type = 'EQ'

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
        Initializes the EqualityConstraintSearcher with the given NLP model and parameters.

        :param nlp: An instance of a spaCy Language model used for processing text.
        :param parameters: A dictionary containing settings and parameters for equality constraint searching.
        """
        super().__init__(nlp, parameters)
        self.match_type = 'META'

    def create_patterns(self, dict):
        """
        Creates patterns for meta constraints based on the dictionary.

        This method generates spaCy patterns that are used to identify meta constraints, for example those triggered by enumeration or if-clauses.

        :param dict: A dictionary with phrases.
        :return: A dictionary of spaCy patterns for identifying equality constraints.
        """
        patterns = {}
        for phrase in dict.keys():
            phrase_pattern = [{"LOWER": word} for word in phrase.split()]
            patterns[phrase.upper().replace(" ", "_")] = phrase_pattern
        return patterns

def search_constraints(nlp, text, equality_params, inequality_params, 
inequality_exception_patterns, equality_exception_patterns, id=1):
    """
    Conducts a combined search for both equality and inequality constraints within a given text.

    This function initializes both InequalityConstraintSearcher and EqualityConstraintSearcher, performs the search, and then combines their findings.

    :param nlp: A spaCy Language model instance used for text processing.
    :param text: The text to search within for constraints.
    :param equality_params: Parameters for the EqualityConstraintSearcher.
    :param inequality_params: Parameters for the InequalityConstraintSearcher.
    :param inequality_exception_patterns: Exception patterns for the InequalityConstraintSearcher.
    :param equality_exception_patterns: Exception patterns for the EqualityConstraintSearcher.
    :param id: The ID of the first match found by the current function within the document.
    :return constraints: A dictionary with detailed information about the found constraints. 
    :return id: The ID of the last match + 1 found by the current function within the document.
    """
    # Initialize searchers
    inequality_searcher = InequalityConstraintSearcher(nlp, inequality_params)
    equality_searcher = EqualityConstraintSearcher(nlp, equality_params)

    # Perform the searches for inequality matches and overwrite ID
    inequality_match_details, id = inequality_searcher.search_matches(text, inequality_exception_patterns, id)
    # Perform the searches for equality matches adn overwrite ID
    equality_match_details, id = equality_searcher.search_matches(text, equality_exception_patterns, id)

    # Combine matches, types, negations
    combined_matches = inequality_match_details['match'] + equality_match_details['match']
    combined_match_types = [inequality_searcher.match_type] * len(inequality_match_details['match']) + \
                           [equality_searcher.match_type] * len(equality_match_details['match'])
    combined_negations = [negation[1] for negation in inequality_match_details['negation'] if negation[0]] + \
                         [negation[1] for negation in equality_match_details['negation'] if negation[0]]

    # Create output dict
    constraints = {
        'ID': inequality_match_details['id'] + equality_match_details['id'],
        'Type': combined_match_types,
        'Match': combined_matches,
        'Pattern': inequality_match_details['patterns'] + equality_match_details['patterns'],
        'Exception': inequality_match_details['exception'] + equality_match_details['exception'],
        'Negation': [neg[0] for neg in inequality_match_details['negation']] + [neg[0] for neg in equality_match_details['negation']],
        'Predecessor': inequality_match_details['predecessor'] + equality_match_details['predecessor'],
        'Successor': inequality_match_details['successor'] + equality_match_details['successor'],
    }

    # Highlighting text
    ConstraintSearcher(nlp, equality_params).highlight_matches(text, combined_matches, combined_negations, combined_match_types)

    # Print in tabular format
    combined_table_data = zip(*constraints.values())
    print(tabulate(combined_table_data, headers=constraints.keys()))

    return constraints, id

def search_constraints_in_data(nlp, data, equality_params, inequality_params, 
inequality_exception_patterns, equality_exception_patterns):
    """
    Conducts a combined search for both equality and inequality constraints for all use cases.

    This function initializes both InequalityConstraintSearcher and EqualityConstraintSearcher, performs the search, and then combines their findings.

    Wrapper function for search_constraints(). 

    :param nlp: A spaCy Language model instance used for text processing.
    :param data: A dictionary with one dataframe per use case containing pre-processed chunks of text in the 'Processed' column.  
    :param equality_params: Parameters for the EqualityConstraintSearcher.
    :param inequality_params: Parameters for the InequalityConstraintSearcher.
    :param inequality_exception_patterns: Exception patterns for the InequalityConstraintSearcher.
    :param equality_exception_patterns: Exception patterns for the EqualityConstraintSearcher.
    :return constraints: A dictionary with detailed information about the found constraints. 
    """
    constraints_tmp = defaultdict(lambda: defaultdict(list)) # Enable dict assignment before knowing the keys

    for use_case, df in data.items():
        print("-"*40)
        print("-"*40, "\n")
        print(use_case.upper())
        print("-"*40)
        print("-"*40, "\n")

        id = 1  # Initialize ID for each use case
        for index, row in df.iterrows(): # Iterate over the rows in the dataframe
            for chunk_index, chunk in enumerate(row['Processed']):  # Iterate over each chunk in the Processed column
                print("\n###CHUNK###\n")
                new_constraints, id = search_constraints(nlp, chunk, equality_params, inequality_params, 
                                                            inequality_exception_patterns, equality_exception_patterns, id)
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