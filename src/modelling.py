from spacy.matcher import Matcher
from termcolor import colored
from tabulate import tabulate

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
        self.unique_matches = {'type': [], 'matches': [], 'patterns': [], 'exception': [], 'negation': [], 'symbol': []}
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

    def search_matches(self, text, exception_patterns):
        """
        Searches for matches in the text based on defined patterns, considering exceptions.

        :param text: The text in which to search for matches.
        :param exception_patterns: A dictionary of patterns that should be treated as exceptions.
        :return: A dictionary containing details of the unique matches found in the text.
        """
        expanded_dict = self.expand_dict(self.parameters["general"])
        patterns = self.create_patterns(expanded_dict)
        formatted_exception_patterns = {key.upper().replace(" ", "_"): value for key, value in exception_patterns.items()}
        for key, value in formatted_exception_patterns.items():
            if callable(value):
                patterns[key] = value()
            else:
                patterns[key] = value
        matcher = Matcher(self.nlp.vocab)
        for key, pattern in patterns.items():
            matcher.add(key, [pattern])
        doc = self.nlp(text)
        matches = matcher(doc)
        seen_tokens = set()
        for match_id, start, end in matches:
            if end not in seen_tokens:
                match_str = self.nlp.vocab.strings[match_id]
                formatted_match_str = match_str.upper().replace(" ", "_")
                is_negated = self.check_for_negation(doc, start, self.parameters["negation_tokens"], self.parameters["window_size"])
                negated_token = start - 1 if is_negated else None
                is_exception = formatted_match_str in formatted_exception_patterns
                self.unique_matches['type'].append(self.match_type)
                self.unique_matches['matches'].append((start, end))
                self.unique_matches['patterns'].append(formatted_match_str)
                self.unique_matches['exception'].append(is_exception)
                self.unique_matches['negation'].append((is_negated, negated_token))
                phrase = match_str.replace("_", " ").lower()
                symbol = expanded_dict.get(phrase, None)
                if symbol is not None and is_negated:
                    symbol = self.parameters["negation_operators"].get(symbol, symbol)
                self.unique_matches['symbol'].append(symbol)
                seen_tokens.add(end)
        return self.unique_matches

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

def search_constraints(nlp, text, equality_params, inequality_params, 
inequality_exception_patterns, equality_exception_patterns):
    """
    Conducts a combined search for both equality and inequality constraints within a given text.

    This function initializes both InequalityConstraintSearcher and EqualityConstraintSearcher, performs the search, and then combines their findings.

    :param nlp: A spaCy Language model instance used for text processing.
    :param text: The text to search within for constraints.
    :param equality_params: Parameters for the EqualityConstraintSearcher.
    :param inequality_params: Parameters for the InequalityConstraintSearcher.
    :param inequality_exception_patterns: Exception patterns for the InequalityConstraintSearcher.
    :param equality_exception_patterns: Exception patterns for the EqualityConstraintSearcher.
    :return: None. The function prints a tabulated summary of the combined constraint search results.
    """
    # Initialize searchers
    inequality_searcher = InequalityConstraintSearcher(nlp, inequality_params)
    equality_searcher = EqualityConstraintSearcher(nlp, equality_params)

    # Perform the searches
    inequality_match_details = inequality_searcher.search_matches(text, inequality_exception_patterns)
    equality_match_details = equality_searcher.search_matches(text, equality_exception_patterns)

    # Combine matches and types
    combined_matches = inequality_match_details['matches'] + equality_match_details['matches']
    combined_match_types = [inequality_searcher.match_type] * len(inequality_match_details['matches']) + \
                           [equality_searcher.match_type] * len(equality_match_details['matches'])
    combined_negations = [negation[1] for negation in inequality_match_details['negation'] if negation[0]] + \
                         [negation[1] for negation in equality_match_details['negation'] if negation[0]]

    # Highlighting Text
    ConstraintSearcher(nlp, equality_params).highlight_matches(text, combined_matches, combined_negations, combined_match_types)

    # Combined Tabular Data
    combined_table_data = zip(
        combined_match_types,
        combined_matches,
        inequality_match_details['patterns'] + equality_match_details['patterns'],
        inequality_match_details['exception'] + equality_match_details['exception'],
        [neg[0] for neg in inequality_match_details['negation']] + [neg[0] for neg in equality_match_details['negation']],
        inequality_match_details['symbol'] + equality_match_details['symbol']
    )
    print(tabulate(combined_table_data, headers=["Type", "Match", "Pattern", "Exception", "Negation", "Symbol"]))