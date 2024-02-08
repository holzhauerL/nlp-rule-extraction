import random
from collections import defaultdict
from tabulate import tabulate
from termcolor import colored
import pandas as pd
from spacy.matcher import Matcher
from spacy import displacy
from openai import OpenAI

class ConstraintSearcher:
    """
    Base class for searching specific constraint patterns in text using spaCy's Matcher. 
    Provides a framework to define and search for patterns related to constraints like inequalities or equalities. 
    """
    def __init__(self, nlp, parameters):
        """
        Initializes the ConstraintSearcher.

        :param nlp: An instance of a spaCy Language model used for processing text.
        :param parameters: A dictionary containing various settings and parameters for constraint searching.
        """
        self.nlp = nlp
        self.parameters = parameters
        self.constraints = {'id': [], 'type': [], 'match': [], 'pattern': [], 'exception': [], 'negation': [], 'symbol': [],  'level': [], 'successor': [], 'context': []}
        self.type = None
        self.empty = "NA"
        self.con_follow = "FOLLOW"
        self.con_start = "START"
        self.con_end = "END"
        self.con_and = "AND"
        self.con_or = "OR"

    def _add_constraint(self, id, match=None, pattern=None, exception=None, negation=None, symbol=None, level=None, connector_suc=None, context=None):
        """
        Adds a new constraint to the constraints dictionary.

        Each parameter corresponds to a different attribute of a constraint. The method updates the constraints dictionary by appending the provided attribute values to the respective lists.

        :param type: The type of the constraint.
        :param match: The spaCy match.
        :param pattern: Name of the matching pattern.
        :param exception: Indicates if it is an exception.
        :param negation: Indicates if there is a negation in the constraint.
        :param symbol: Symbol representing the constraint.
        :param level: The level of an enumeration constraint.
        :param connector_suc: The connection of the successor to the constraint.
        :param context: The span indicating the context for the ConstraintBuilder.
        """
        self.constraints['id'].append(id)
        self.constraints['type'].append(self.type)
        self.constraints['match'].append(match if match is not None else self.empty)
        self.constraints['pattern'].append(pattern if pattern is not None else self.empty)
        self.constraints['exception'].append(exception if exception is not None else self.empty)
        self.constraints['negation'].append(negation if negation is not None else self.empty)
        self.constraints['symbol'].append(symbol if symbol is not None else self.empty)
        self.constraints['level'].append(level if level is not None else self.empty)
        self.constraints['successor'].append((id+1, connector_suc))
        self.constraints['context'].append(context if context is not None else self.empty)

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

    def highlight_matches(self, text, linebreaks, matches, negations, match_types, patterns):
        """
        Highlights the matches in the text based on their types and negations. 
        Uses different colors for different types of matches and negations. 
        Ensures that a token is highlighted only once.

        :param text: The original text in which matches are to be highlighted.
        :param linebreaks: Information on succeding line breaks for meta constraint searching.
        :param matches: A list of match tuples (start_index, end_index) or ((start_index, end_index), token_index) for 'META' type.
        :param negations: A list of indices where negations occur.
        :param match_types: A list of types corresponding to each match.
        :param patterns: A list of pattern names corresponding to each match.
        :return: None. The function prints the highlighted text.
        """
        doc = self.nlp(text)
        highlighted_text = ""
        last_index = 0
        combined = []
        highlighted_indices = set()  # Set to keep track of highlighted tokens
        linebreak_indices = [] # Indices of characters with succeeding linebreaks
        for token in doc:
            try:
                if linebreaks[token.i]:
                    linebreak_indices.append(doc[token.i].idx + len(doc[token.i].text))
            except:
                pass # To prevent errors due to a non-optimal tokenisation (example: 1,2 counted as one token due to missing space)

        # Remove duplicates and None elements
        negations = [n for n in set(negations) if n is not None]

        for match, mtype, pattern in zip(matches, match_types, patterns):
            if isinstance(match[0], tuple):  # ((start, end_enum), end)
                range_match, single_token = match
                combined.append((range_match[0], range_match[1] + 1, 'match', mtype, pattern))
                combined.append((match[1], match[1] + 1, 'match', mtype, pattern))
            elif match[1] - match[0] > 1:  # (start, end)
                combined.append((match[0], match[0] + 1, 'match', mtype, pattern))
                combined.append((match[1], match[1] + 1, 'match', mtype, pattern))
            else:
                combined.append((match[0], match[1] + 1, 'match', mtype, pattern))

        combined.extend([(negation, negation + 1, 'negation', None, None) for negation in negations])
        combined.sort(key=lambda x: x[0])

        for start, end, type, mtype, pattern in combined:

            if start in highlighted_indices:
                continue  # Skip if start index is already highlighted

            start_char = doc[start].idx
            end_char = doc[end - 1].idx + len(doc[end - 1].text)

            # Add text before the current match/negation
            while True:
                if len(linebreak_indices):
                    if linebreak_indices[0] <= start_char:
                        highlighted_text += text[last_index:linebreak_indices[0]]
                        highlighted_text += "\n"
                        last_index = linebreak_indices[0]
                        linebreak_indices.pop(0)
                    else:
                        break
                else:
                    break

            highlighted_text += text[last_index:start_char]

            # Determine color based on type and match type
            color = 'magenta'  # Default color for other types
            if type == 'match':
                if mtype == 'INEQ':
                    color = 'green'
                elif mtype == 'EQ':
                    color = 'blue'
                elif mtype == 'META' and "CONNECTOR" not in pattern:
                    color = 'yellow'
                elif mtype == 'META':
                    color = 'magenta'
            elif type == 'negation':
                color = 'red'

            highlighted_text += colored(text[start_char:end_char], color, attrs=['bold'])
            last_index = end_char

            # Update highlighted indices
            for i in range(start, end):
                highlighted_indices.add(i)

        highlighted_text += text[last_index:]
        print(highlighted_text)

    def search_constraint_items(self, text, id):
        """
        Searches for constraint items in the text based on defined patterns, considering exceptions.

        :param text: The text in which to search for constraint items.
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
                is_exception = formatted_match_str in formatted_exception_patterns or formatted_match_str in {key.upper().replace(" ", "_") for key in exceptions_dict.keys()}

                # Handling symbols
                phrase = match_str.replace("_", " ").lower()
                symbol = merged_dict.get(phrase, None)
                if symbol is not None and is_negated:
                    symbol = self.parameters["negation_operators"].get(symbol, symbol)
        
                seen_tokens.add(end)

                # Add constraint
                self._add_constraint(id, match=(start, end-1), pattern=formatted_match_str, exception=is_exception, negation=(negated_token, is_negated), symbol=symbol, connector_suc=self.con_follow)

                # Updating the ID
                id += 1

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
            phrase_pattern.extend([{"TEXT": {"NOT_IN": [".", ","]}, "IS_DIGIT": False, "OP": "*"},{"LIKE_NUM": True}])
            patterns[phrase] = phrase_pattern
            # # Account for float numbers
            # phrase_pattern_float = phrase_pattern.extend([{"TEXT": {"IN": ["."]}},{"IS_DIGIT": True}])
            # patterns[phrase + "_float"] = phrase_pattern_float
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
    """
    def __init__(self, nlp, parameters):
        """
        Initializes the MetaConstraintSearcher with the given NLP model and parameters.

        :param nlp: An instance of a spaCy Language model used for processing text.
        :param parameters: A dictionary containing settings and parameters for meta constraint searching.
        """
        super().__init__(nlp, parameters)
        self.type = 'META'

    def search_enum_constraint_items(self, text, enumeration_summary, id):
        """
        Searches and processes enumeration constraint items based on enumeration summaries.

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
                logical_connections[level] = self.con_and
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
                    logical_connections[level] = self.con_or
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
            

            # Set successor
            connector_suc = logical_connections[level]

            # Add constraint
            self._add_constraint(id, match=((start_token_enumeration, end_token_enumeration),end_token_item), pattern=pattern_names[level], exception=exceptions[level], level=level, connector_suc=connector_suc)

            id += 1

            last_level = level

        return id

    def search_if_clauses(self, text, succeeding_linebreaks, id):
        """
        Searches for 'if' clauses within a single line of text and records their details.

        This method uses spaCy pattern matching to identify 'if' structures within the text. 
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
            matcher.add(pattern_name, [pattern_func])

        matches = matcher(doc)
        processed_matches = {}

        for match_id, start, end in matches:
            if not succeeding_linebreaks[start:end].count(True):  # Ensure match is within a single line
                # reduce the match end to the last comma or 'then'
                reduced_end = end
                for i in range(end, start, -1):
                    if doc[i].lower_ == ',' or doc[i].lower_ == 'then':
                        reduced_end = i
                        break
                
                negated_token, is_negated = self.check_for_negation(doc, start, self.parameters["negation_tokens"], self.parameters["window_size"])

                if start not in processed_matches or (reduced_end - start) > (processed_matches[start][1] - processed_matches[start][0]):
                    processed_matches[start] = (start, reduced_end)

        for start, (start, end) in processed_matches.items():

            # Add constraint
            self._add_constraint(id, match=(start, end), pattern=self.nlp.vocab.strings[match_id].upper(), exception=False, negation=(negated_token, is_negated),connector_suc=self.con_and)

            id += 1

        return id

    def search_for_clauses(self, text, succeeding_linebreaks, id):
        """
        Searches for 'for' clauses within a single line of text and records their details.

        :param text: The text in which to search for 'for' clauses.
        :param succeeding_linebreaks: A list indicating linebreaks in the text.
        :param id: The ID to start with for the first 'for' clause found.
        :return: The updated ID.
        """
        doc = self.nlp(text)
        matcher = Matcher(self.nlp.vocab)
        for pattern_name, pattern_func in self.parameters['for_patterns'].items():
            matcher.add(pattern_name, [pattern_func])

        matches = matcher(doc)
        processed_matches = {}

        for match_id, start, end in matches:
            if not succeeding_linebreaks[start:end].count(True):  # Ensure match is within a single lines
                reduced_end = end
                for i in range(end, start, -1):
                    if doc[i].lower_ == ':' or doc[i].lower_ == ',':
                        reduced_end = i
                        break

                if start not in processed_matches or (reduced_end - start) > (processed_matches[start][1] - processed_matches[start][0]):
                    processed_matches[start] = (start, reduced_end)

        for start, (start, reduced_end) in processed_matches.items():

            # Add constraint
            self._add_constraint(id, match=(start, reduced_end), pattern=self.nlp.vocab.strings[match_id].upper(), exception=False, connector_suc=self.con_and)

            id += 1

        return id
    
    def _dict_to_df(self, constraints, columns_to_keep = ['id', 'type', 'match_start', 'match_end', 'pattern', 'level']):
        """
        Convertes the constraints dictionary into a pandas DataFrame, while also converting certain types, imputing values and dropping columns.

        :param constraints: Dictionary with the constraints.
        :return: DataFrame with the constraints.
        """
        constraints_df = pd.DataFrame.from_dict(constraints)

        if 'match_start' in columns_to_keep:
            constraints_df['match_start'] = constraints_df['match'].apply(lambda x: x[0][0] if isinstance(x[0], tuple) else x[0]).astype(int)

        if 'enum_item_end' in columns_to_keep:
            constraints_df['enum_item_end'] = constraints_df['match'].apply(lambda x: x[0][1] if isinstance(x[0], tuple) else 0).astype(int)
        
        if 'match_end' in columns_to_keep:
            constraints_df['match_end'] = constraints_df['match'].apply(lambda x: x[1]).astype(int)

        if 'level' in columns_to_keep:
            constraints_df['level'] = constraints_df['level'].replace({'NA': 0}).astype(int)

        if 'context_start' in columns_to_keep:
            constraints_df['context_start'] = constraints_df['context'].apply(lambda x: x[0]).astype(int)
        
        if 'context_end' in columns_to_keep:
            constraints_df['context_end'] = constraints_df['context'].apply(lambda x: x[1]).astype(int)

        if 'negation' in columns_to_keep:
            constraints_df['negation'] = constraints_df['negation'].apply(lambda x: x[0]if isinstance(x[0], int) else 0).astype(int)

        # Keep only required columns
        constraints_df = constraints_df[columns_to_keep]

        return constraints_df

    def determine_context(self, text, constraints, linebreaks):
        """
        Determines the context for each constraint, based on their type.

        :param text: The text to search within for constraints.
        :param constraints: A dictionary with detailed information about the found constraints.
        :param linebreaks: Boolean array indicating line breaks after each token.
        :return: The input dictionary, enhanced with context.
        """
        doc = self.nlp(text)
        doc_start = 0
        doc_end = len(doc) - 1
        context_limits = set(self.parameters["context_limits"])

        # Create a DataFrame with the subset of the columns necessary for the checks to reduce runtime
        constraints_df = self._dict_to_df(constraints)

        # Consider only constraints which are of type str (still empty)
        idx = [i for i in range(len(constraints['id'])) if isinstance(constraints['context'][i],str)]

        for index, row in constraints_df.loc[idx].iterrows():

            # Reset context start and end
            context_start = doc_start
            context_end = doc_end

            # Equality and inequality constraints
            if row['type'] in ['INEQ', 'EQ']:
                # Find the encompassing constraints
                encompassing_constraints = constraints_df[
                    (constraints_df['match_start'] <= row['match_start']) &
                    (constraints_df['match_end'] >= row['match_end']) &
                    (constraints_df.index != index)
                ].copy()
                # Find the one with minimum range difference
                if not encompassing_constraints.empty:
                    encompassing_constraints['range_diff'] = encompassing_constraints.apply(
                        lambda x: abs(x['match_start'] - row['match_start']) + abs(x['match_end'] - row['match_end']),
                        axis=1
                    )
                    min_diff_constraint = encompassing_constraints.loc[encompassing_constraints['range_diff'].idxmin()]
                    # Adjust context based on the found constraint
                    context_start_encom = [min_diff_constraint['match_start']]
                    context_end_encom = [min_diff_constraint['match_end']]
                else:
                    context_start_encom = context_end_encom = []

                # Determine context_start
                context_start_options = [i for i in range(row['match_start'] - 1, -1, -1) if doc[i].text in context_limits]
                context_start_other_constraints = [constraints_df.at[j, 'match_end'] for j in range(len(constraints_df)) if j != index and constraints_df.at[j, 'match_end'] < row['match_start']]
                context_start_linebreaks = [i+1 for i, lb in enumerate(linebreaks) if lb and i < row['match_start']]
                context_start = max(context_start_options + context_start_other_constraints + context_start_linebreaks + context_start_encom, default=doc_start)

                # Determine context_end
                context_end_options = [i for i in range(row['match_end'] + 1, len(doc)) if doc[i].text in context_limits]
                context_end_other_constraints = [constraints_df.at[j, 'match_start'] for j in range(len(constraints_df)) if j != index and constraints_df.at[j, 'match_start'] > row['match_end']]
                context_end_linebreaks = [i for i, lb in enumerate(linebreaks) if lb and i >= row['match_end']]
                context_end = min(context_end_options + context_end_other_constraints + context_end_linebreaks + context_end_encom, default=doc_end)
            # All meta constraints except for enumerations
            else:

                match_start, match_end = row['match_start'], row['match_end']

                # Context start for META is the match start
                context_start = match_start

                if "ENUM" not in row['pattern']:
                    # Determine context_end for non-enumeration meta constraints
                    context_end_options = [i for i in range(row['match_end'] + 1, len(doc)) if doc[i].text in context_limits]
                    context_end_other_constraints = [constraints_df.at[j, 'match_start'] for j in range(len(constraints_df)) if j != index and constraints_df.at[j, 'match_start'] > row['match_end']]
                    context_end_linebreaks = [i for i, lb in enumerate(linebreaks) if lb and i >= row['match_end']]
                    context_end = min(context_end_options + context_end_other_constraints + context_end_linebreaks, default=doc_end)
                else:
                    context_end = match_end

            constraints['context'][index] = (context_start, context_end)

        return constraints
    
    def _update_ids(self, constraints, first_id):
        """ 
        Updates the ids in 'id' and 'successor' fields of the constraints dict.

        :param constraints: A dictionary with detailed information about the found constraints.
        :param first_id: The id to start with.
        :return: A dictionary with detailed information about the updated constraints.
        """

        # Update the IDs to match the new structure
        last_id = first_id + len(constraints['id']) - 1
        constraints['id'] = list(range(first_id, last_id + 1))

        # Update successor IDs
        for i in range(len(constraints['successor'])):
            constraints['successor'][i] = (constraints['id'][i]+1, constraints['successor'][i][1])
        
        return constraints
    
    def insert_connections(self, text, constraints):
        """
        Searches and inserts connections in between constraints.
        
        :param text: The text to search within for constraints.
        :param constraints: A dictionary with detailed information about the found constraints.
        :return: A dictionary with detailed information about the ranked and connected constraints.
        """
        first_id = constraints['id'][0]

        # Create a DataFrame with the subset of the columns necessary for the checks to reduce runtime
        constraints_df = self._dict_to_df(constraints, columns_to_keep = ['id', 'type', 'match_start', 'match_end', 'enum_item_end', 'pattern', 'level', 'context_start', 'context_end'])

        # Create the idx_map dictionary
        idx_map = {old_idx: old_idx for old_idx in constraints_df.index}
        
        # Creating connector patterns
        patterns = {}
        for phrase in self.parameters["connectors"].keys():
            phrase_pattern = [{"LOWER": word} for word in phrase.split()]
            patterns[phrase.upper().replace(" ", "_")] = phrase_pattern

        exception_patterns = []

        # Adding connector exception patterns
        for key, value in self.parameters["connector_exception_pattern"].items():
            phrase_pattern = [{"LOWER": word} for word in value[0].split()]
            exception_patterns.append(key.upper().replace(" ", "_"))
            patterns[exception_patterns[-1]] = phrase_pattern

        # Matching patterns
        matcher = Matcher(self.nlp.vocab)
        for key, pattern in patterns.items():
            matcher.add(key, [pattern])
        
        doc = self.nlp(text)
        matches = matcher(doc)

        for match_id, start, end in matches:

            # Default connectors
            if self.nlp.vocab.strings[match_id] not in exception_patterns:
                exception = False
                connector = self.parameters["connectors"][self.nlp.vocab.strings[match_id].lower().replace("_", " ")]

                subset_idx_left = constraints_df[(constraints_df['match_end'] <= start) & (constraints_df['context_end'] >= end)].index

                subset_idx_right = constraints_df[(constraints_df['match_start'] >= end)].index

                # Check if subset_idx_left is not empty, then find the index for which the corresponding match_end is the highest
                if not subset_idx_left.empty :
                    max_end_idx = constraints_df.loc[subset_idx_left, 'match_end'].idxmax()
                    if max_end_idx in subset_idx_left:
                        left_idx = max_end_idx
                    else:
                        left_idx = None
                else:
                    left_idx = None

                # Check if subset_idx_right is not empty, then find the index for which the corresponding match_start is the lowest
                if not subset_idx_right.empty:
                    min_start_idx = constraints_df.loc[subset_idx_right, 'match_start'].idxmin()
                    if min_start_idx in subset_idx_right:
                        right_idx = min_start_idx
                    else:
                        right_idx = None
                else:
                    right_idx = None
            # Exception
            else:
                exception = True
                connector = self.parameters["connector_exception_pattern"][self.nlp.vocab.strings[match_id].lower().replace("_", " ")][1]
                # Check if there is an encompassing enumeration constraint, but make sure that the found connector is not part of the enumeration item or that it is at the end of the enumeration constraint
                enum_idx = constraints_df[(constraints_df['context_start'] < start) & (constraints_df['context_end'] > end) & (constraints_df['pattern'].str.contains("ENUM")) & (start > constraints_df['enum_item_end'])].index
                # Only consider those exceptions within the context of an enumeration item
                if not enum_idx.empty:
                    left_idx = right_idx = enum_idx[0]
                else:
                    left_idx = right_idx = None

            # Check if both left_idx and right_idx are not None
            if left_idx is not None and right_idx is not None:
                new_idx = idx_map[left_idx] + 1

                # Insert new list element at position after new_idx
                new_element = {'id': 99, 'type': self.type, 'match': (start, end - 1), 'pattern': 'CONNECTOR_' + self.nlp.vocab.strings[match_id], 'exception': exception, 'level': self.empty,'successor': (right_idx, connector), 'context': (constraints_df.at[left_idx, 'context_start'], constraints_df.at[right_idx, 'context_end'])}

                for key in constraints.keys():
                    if key in new_element.keys():
                        value = new_element[key]
                    else:
                        value = self.empty
                    constraints[key].insert(new_idx+1, value)

                # Update idx_map
                for key, value in idx_map.items():
                    if value > new_idx:
                        idx_map[key] = value + 1

        return self._update_ids(constraints, first_id)
    
    def _insert_new_element(self, new_elements, constraints, idx_map):
        """
        Helper function to insert a new element into the existing set of constraints and update the index map.

        :param new_elements: Array with the new elements to insert.
        :param constraints: A dictionary with detailed information about the found constraints.
        :param idx_map: Mapping the old indices to the new ones.
        :return constraints: Updated constraints dict.
        :return idx_map: Updated index map.
        """
        # Insert new BOOL constraints
        for index, new_element in sorted(new_elements, key=lambda x: x[0], reverse=True):
            new_idx = idx_map[index] + 1
            for key in constraints:
                if key in new_element:
                    constraints[key].insert(new_idx, new_element[key])
                else:
                    constraints[key].insert(new_idx, self.empty)

            # Update idx_map for subsequent inserts
            for key, value in idx_map.items():
                if value >= new_idx:
                    idx_map[key] = value + 1
        return constraints, idx_map

    def insert_bool(self, constraints):
        """
        Inserts constraints of type 'BOOL' into existing constraints dictionary.
        
        :param constraints: A dictionary with detailed information about the found constraints.
        :return: A dictionary with the inserted 'BOOL' constraints.
        """
        first_id = constraints['id'][0]

        # Create a DataFrame with the subset of the columns necessary for the checks to reduce runtime
        constraints_df = self._dict_to_df(constraints, columns_to_keep = ['id', 'type', 'match_start', 'match_end', 'pattern', 'level', 'context_start', 'context_end'])

        # Create the idx_map dictionary
        idx_map = {old_idx: old_idx for old_idx in constraints_df.index}

        new_elements = []

        # Iterate over rows of DataFrame
        for index, row in constraints_df.iterrows():
            
            if any(x in row['pattern'] for x in ['ENUM', 'FOR_', 'IF_']):
                # Check for encommpassed, non-connector constraints 
                subset_indices = constraints_df[(constraints_df['match_start'] > row['match_start']) & (constraints_df['match_end'] < row['match_end']) & (~constraints_df['pattern'].str.contains("CONNECTOR"))].index
                # If no constraint found within an enumeration item, make it a boolean one
                if len(subset_indices) == 0:
                    new_idx = idx_map[index]
                    if 'ENUM' in row['pattern']:
                        ((x,y),z) = constraints['match'][new_idx]
                    else:
                        y,z = constraints['match'][new_idx]
                    
                    match_start = y + 1
                    match_end = z - 1

                    if match_start > match_end:
                        tmp = match_start
                        match_start = match_end
                        match_end = tmp

                    # Insert new list element at position after new_idx
                    new_element = {'id': 99, 'type': self.type, 'match': (match_start, match_end), 'pattern': 'BOOL', 'exception': False, 'symbol': "==",'level': self.empty, 'successor': (100, self.con_follow), 'context': (match_start, match_end)}
                    
                    for key in constraints.keys():
                        if key in new_element.keys():
                            value = new_element[key]
                        else:
                            value = self.empty
                        constraints[key].insert(new_idx+1, value)

                    # Update idx_map
                    for key, value in idx_map.items():
                        if value > new_idx:
                            idx_map[key] = value + 1
            
            # Adding boolean constraints after a for/if constraint
            if any(x in row['pattern'] for x in ['FOR_', 'IF_']):
                match_start_bool = row['match_end'] + 1
                # Find the closest match_start greater than match_start_bool
                potential_starts = constraints_df[(constraints_df['match_start'] > match_start_bool)]['match_start']
                potential_starts_meta = constraints_df[(constraints_df['match_start'] > match_start_bool) & (constraints_df['type'] == 'META')]['match_start']

                if not potential_starts_meta.empty:
                    match_end_bool = potential_starts_meta.min() - 1
                elif potential_starts.empty:
                    match_end_bool = row['context_end']
                else:
                    match_end_bool = None

                if match_end_bool and match_start_bool <= match_end_bool:
                    # Generate new BOOL constraint
                    new_element = {'id': 99, 'type': self.type, 'match': (match_start_bool, match_end_bool), 'pattern': 'BOOL', 'exception': False, 'symbol': "==",'level': self.empty, 'successor': (100, self.con_follow), 'context': (match_start_bool, match_end_bool)}
                    new_elements.append((index, new_element))

        # Insert new BOOL constraints
        constraints, idx_map = self._insert_new_element(new_elements, constraints, idx_map)

        # Recreate the DataFrame with the updated constraints, now also including the added BOOL constraints
        constraints_df = self._dict_to_df(constraints, columns_to_keep = ['id', 'type', 'match_start', 'match_end', 'pattern', 'level', 'context_start', 'context_end'])

        # Create the idx_map dictionary
        idx_map = {old_idx: old_idx for old_idx in constraints_df.index}

        new_elements = []

        # Insert addtional BOOL, if CONNECTOR in between
        for index, row in constraints_df.iterrows():

            if 'CONNECTOR' in row["pattern"]:
                encompassing_bool_index = constraints_df[(constraints_df['pattern'] == 'BOOL') & (constraints_df['match_start'] < row['match_start']) & (constraints_df['match_end'] > row['match_end'])].index

                if not encompassing_bool_index.empty:
                    for idx in encompassing_bool_index:
                        # Generate new BOOL
                        new_element = {'id': 99, 'type': self.type, 'match': (row['match_end']+1, constraints['match'][idx][1]), 'pattern': 'BOOL', 'exception': False, 'symbol': "==",'level': self.empty, 'successor': (100, self.con_follow), 'context': (row['match_end']+1, constraints['match'][idx][1])}
                        new_elements.append((index, new_element))
                        # Update match and context for encompassing BOOL (only one entry in index array)
                        constraints['match'][idx] = constraints['context'][idx] = (constraints['context'][idx][0], row['match_start']-1)

        # Insert new BOOL constraints
        constraints, idx_map = self._insert_new_element(new_elements, constraints, idx_map)

        return self._update_ids(constraints, first_id)
    
    def sort_and_prune(self, constraints):
        """
        Sorts and prunes the constraints.
        
        :param constraints: A dictionary with detailed information about the found constraints.
        :return: A dictionary with detailed information about the sorted and pruned constraints.
        """
        first_id = constraints['id'][0]

        # Create a DataFrame with the subset of the columns necessary for the checks to reduce runtime
        constraints_df = self._dict_to_df(constraints, columns_to_keep = ['id', 'type', 'match_start', 'match_end', 'pattern', 'level', 'context_start', 'context_end', 'negation'])
        
        # Sort the constraints by match_start in ascending order
        constraints_df.sort_values(by='match_start', inplace=True)

        # Dropping constraints based on specified conditions
        rows_to_drop = []
        previous_row = None
        for index, row in constraints_df.iterrows():

            # Check if a 'BOOL' constraint is encompassing a 'EQ' or 'INEQ' constraint
            if row['pattern'] == 'BOOL':
                matching_indices = constraints_df[((constraints_df['type'] == 'EQ') | (constraints_df['type'] == 'INEQ')) & (row['match_start'] <= constraints_df['match_start']) &  (row['match_end'] >= constraints_df['match_end'])].index
                if not matching_indices.empty:
                    rows_to_drop.append(index)

            # If it is a 'FOR' or 'IF' constraint and there are no enumeration constraints in this chunk 
            if any(x in row['pattern'] for x in ['FOR_', 'IF_']) and not (constraints_df['pattern'].str.contains("ENUM").any()):
                # Drop if there are no 'EQ' or 'INEQ' constraints within the context (incl. the borders) 
                subset_df = constraints_df[(constraints_df['match_start'] >= row['context_start']) & (constraints_df['match_start'] <= row['context_end']) & (constraints_df['type'] != 'META') & (~constraints_df.index.isin([index]))]
                if subset_df.empty:
                    rows_to_drop.append(index)
                    # Find encompassed BOOL constraints to drop
                    bool_indices = constraints_df[(constraints_df['pattern'] == 'BOOL') & (constraints_df['match_end'] <= row['context_end']) & (constraints_df['match_start'] >= row['context_start'])].index
                    rows_to_drop.extend(bool_indices)
                 
            if row['type'] in ['EQ', 'INEQ'] and previous_row is not None:
                # Drop last 'EQ' or 'INEQ' if two subsequent constraints of the same type; also check context (to account for linebreaks) and negation
                if previous_row['type'] == row['type'] and previous_row['context_end'] == row['match_start'] and row['negation'] == 0:
                    rows_to_drop.append(index)
                    # Update context end for remaining constraint
                    constraints['context'][previous_index] = (constraints['context'][previous_index][0], row['context_end'])

                # Drop 'EQ' if subsequent 'INEQ'; also check context (to account for linebreaks)
                elif previous_row['type'] == 'EQ' and row['type'] == 'INEQ' and previous_row['context_end'] == row['match_start']:
                    rows_to_drop.append(previous_index)
                    # Update context start for remaining 'INEQ' constraint
                    constraints['context'][index] = (previous_row['context_start'], constraints['context'][index][1])

            if 'CONNECTOR' in row['pattern']:
                # Drop two subsequent 'AND' connectors
                if '_AND' in row['pattern'] and '_AND' in previous_row['pattern']:
                    rows_to_drop.append(previous_index)
                    rows_to_drop.append(index)
                    
            # Save the previous value
            previous_row, previous_index = row, index
        
        # Drop the identified rows
        constraints_df = constraints_df.drop(list(set(rows_to_drop)))

        # Identify duplicates based on 'pattern', 'match_start', and 'match_end'
        duplicate_rows = constraints_df.duplicated(subset=['pattern', 'match_start', 'match_end'], keep='first')

        # Drop the identified duplicate rows
        constraints_df = constraints_df[~duplicate_rows]
        
        for key in constraints:
            constraints[key] = [constraints[key][idx] for idx in constraints_df.index]

        return self._update_ids(constraints, first_id)
    
class ConstraintBuilder:
    """
    Class for constraint building based on the constraint items.
    """
    def __init__(self, nlp, parameters, verbose=False):
        """
        Initializes the ConstraintBuilder.

        :param nlp: An instance of a spaCy Language model used for processing text.
        :param parameters: A dictionary containing settings and parameters for the ConstraintBuilder. 
        :param verbose: Parameter to control the visualisation of the dependency tree and the entities. If True, visualisation is generated. 
        """
        self.nlp = nlp
        self.parameters = parameters
        self.verbose = verbose
        self.resources = {'input_tokens':[], 'output_tokens':[], 'input_costs':[], 'output_costs':[]}
        self.con_follow = "FOLLOW"
        self.con_start = "START"
        self.con_end = "END"
        self.con_and = "AND"
        self.con_or = "OR"
        self.empty = "NA"
        self.negations = {
            "<": ">=",
            "<=": ">",
            ">": "<=",
            ">=": "<",
            "==": "!=",
            "!=": "=="
        }

    def build_constraint(self, text, constraints):
        """
        Builds the structure for the formatted constraint for the Gold Standard comparison.

        :param text: The text to search within for constraints.
        :param constraints: A dictionary with detailed information about the found constraints.
        :return formatted_constraints: An array with the formatted constraint.
        """
        random_integer = random.randint(1, 10)
        formatted_constraint = []
        constraint = ""
        step = f"STEP_{random_integer}"
        level_history = []
        number_of_components = len(constraints['id'])
        build_mode = self.parameters["build_mode"]

        i = 0
        open_parentheses = 0
        
        while True:
            type = constraints['type'][i]
            pattern = constraints['pattern'][i]
            level = constraints['level'][i]
            connector = constraints['successor'][i][1]

            # Enumeration items
            if 'ENUM' in pattern:
                # First enumeration item
                if not (level_history):
                    constraint += "(("
                    open_parentheses += 2
                # Not the first item, but same level as before
                elif level == level_history[-1]:
                    constraint += ") " + connector + " (" 
                # Not the first item, one level deeper
                elif level > level_history[-1]:
                    # If the previous constraint component was a connector, do not add a connector
                    if 'CONNECTOR' in constraints['pattern'][i-1]:
                        constraint += "(("
                    else: 
                        constraint += " " + self.con_and + " ((" 
                    open_parentheses += 2
                # Not the first item, one level shallower
                elif level < level_history[-1]:
                    constraint += "))) " + connector + " ("
                    open_parentheses -= 2
                # Save current level
                level_history.append(level)

            # Connectors
            elif 'CONNECTOR' in pattern:
                constraint += " " + connector + " "

            # For-/If-clauses
            elif any(x in pattern for x in ['FOR_', 'IF_']):
                # Special structure for two connected clauses with the same then
                # i + 5 ensures five more constraints after the first for/if
                if i + 5 < number_of_components and 'CONNECTOR' in constraints['pattern'][i+2] and any(x in constraints['pattern'][i+3] for x in ['FOR_', 'IF_']):
                    connector = constraints['successor'][i+2][1]
                    connector_neg = self.con_or if connector == self.con_and else self.con_and 
                    # First, add ELSE (negated IF)
                    constraint += "((" + self.build_component(text, constraints, i+1, build_mode=build_mode, negated=True) + " " + connector_neg + " " + self.build_component(text, constraints, i+4, build_mode=build_mode, negated=True)
                    # Add connector
                    constraint += ") " + self.con_or + " "
                    # Add THEN (IF can be skipped, since connected with OR)
                    constraint += self.build_component(text, constraints, i+5, build_mode=build_mode) + ")"
                    # Skip the already processed items
                    i += 5
                # Default case
                elif i + 2 < number_of_components:
                    # First, add ELSE (negated IF)
                    constraint += "(" + self.build_component(text, constraints, i+1, build_mode=build_mode, negated=True)
                    # Add connector
                    constraint += " " + self.con_or + " "
                    # Add THEN (IF can be skipped, since connected with OR)
                    constraint += self.build_component(text, constraints, i+2, build_mode=build_mode) + ")"
                    # Skip the already processed items
                    i += 2

            # For basic components
            elif type in ['EQ', 'INEQ'] or 'BOOL':
                constraint += self.build_component(text, constraints, i, build_mode=build_mode)

            i += 1
            if i == number_of_components:
                constraint += ")"*open_parentheses
                break

        formatted_constraint.append((step, constraint))

        return formatted_constraint
    
    def _get_subset(self, tokens, indices, filter_stop=True, filter_alpha=True):
        """
        From the list of tokens, retrieve a subset based on indices. Optionally, filter stop words and tokens with specific POS tags.

        :param tokens: List of tokens.
        :param indices: List of indices, referring to tokens.
        :param filter_stop: Flag to determine filtering of stop words. If True, stop words are filtered out. 
        :param filter_alpha: Flag to determine filtering of alpha characters. If True, only alpha characters are considered.
        :return: List of subset of tokens, optionally filtered.
        """
        filter_pos = self.parameters["filter_pos"]
        if filter_stop:
            stop_words = set(self.nlp.Defaults.stop_words)
            subset = " ".join([tokens[idx] for idx in indices if tokens[idx] not in stop_words])
        else:
            subset = " ".join([tokens[idx] for idx in indices])
        
        doc = self.nlp(subset)

        # Filter tokens based on is_alpha and POS tags
        filtered_tokens = [token.text for token in doc if (not filter_alpha or token.is_alpha) and (not filter_pos or token.pos_ in filter_pos)]

        return filtered_tokens

    def build_component(self, text, constraints, i, build_mode='rules', negated=False):
        """
        Building of the constraint components, with the options to do it either rule-based or with a GPT of OpenAI.

        :param text: The text to search within for constraints.
        :param constraints: A dictionary with detailed information about the found constraints.
        :param i: Index in the constraint dict.
        :param build_mode: Determines if the components are build using a rule-based approach ('rules') or with a GPT of OpenAI ('openai').
        :param negated: Flag to determine if a negation should be performed (additional to negations triggered by matches).
        :return: One single constraint component.
        """
        component = ""

        keep_tokens = self.parameters["keep_tokens"]
        estimation_mode = self.parameters["estimation_mode"]
        
        type = constraints['type'][i]
        pattern = constraints['pattern'][i]
        context = constraints['context'][i]
        match = constraints['match'][i]
        symbol = constraints['symbol'][i]

        tokens = text.split()

        # doc = self.nlp(subset_left + " " + subset_right)
        # displacy.render(doc, style='dep', jupyter=True)

        if negated and symbol is not self.empty and build_mode == 'rules':
            symbol = self.negations[symbol]
        
        if build_mode == 'rules':
            # For BOOL
            if 'BOOL' in pattern:
                # To prevent != True
                right_part = "True" if symbol == "==" else "False"
                symbol = "=="

                indices = list(range(match[0],match[1]+1))

                filtered_tokens = self._get_subset(tokens, indices)
                # Going from the end to the beginning, check the number of tokens to keep to the end of doc_left
                filtered_tokens = filtered_tokens[-keep_tokens:]
                # Construct left_part from filtered tokens
                left_part = "_".join(filtered_tokens)

            # Coherent INEQ and EQ patterns
            elif symbol is not self.empty:

                # Define the indices to keep the relation to the original token
                indices_left = list(range(context[0],match[0]))
                indices_right = list(range(match[1]+1,context[1]+1))
                # Get the tokens left and right of the match
                filtered_tokens_left = self._get_subset(tokens, indices_left)
                filtered_tokens_right = self._get_subset(tokens, indices_right)

                # Going from the end to the beginning, check the number of tokens to keep to the end of doc_left
                filtered_tokens_left = filtered_tokens_left[-keep_tokens:] 

                # Coherent INEQ
                if type == 'INEQ':
                    
                    # The reference value
                    right_part = tokens[match[1]]

                    # If the left_array has not yet the number of tokens to keep, use doc_right
                    if len(filtered_tokens_left) < keep_tokens:
                        # Add to the array without exceeding the number of tokens to keep in total
                        additional_tokens_needed = keep_tokens - len(filtered_tokens_left)
                        filtered_tokens_left.extend(filtered_tokens_right[:additional_tokens_needed])

                    filtered_tokens = filtered_tokens_left

                    # Construct left_part from filtered tokens
                    left_part = "_".join(filtered_tokens)

                # Coherent EQ
                if type == 'EQ':
        
                    filtered_tokens_right = filtered_tokens_right[:keep_tokens] 

                    left_part = "_".join(filtered_tokens_left)
                    right_part = "_".join(filtered_tokens_right)

            # TODO: INEQ and EQ exceptions
            else:
                left_part = symbol = right_part = ""

            component = left_part + " " + symbol + " " + right_part

        elif build_mode == 'openai':
            indices = list(range(context[0],context[1]+1))
            search_context = " ".join([tokens[idx] for idx in indices])

            # For BOOL
            if 'BOOL' in pattern:
                key = 'BOOL'
                # To prevent != True
                right_part = "True" if symbol == "==" else "False"
                symbol = "=="
            
            # For coherent INEQ and all EQ patterns
            else:
                key = type
                # Coherent INEQ patterns
                if key == 'INEQ' and symbol is not self.empty:
                    right_part = tokens[match[1]]
                # All EQ
                elif key == 'EQ':
                    right_part = "True" if symbol == "==" else "False"
                    symbol = "=="
                # TODO: INEQ exceptions
                else:
                    key = None
                    component = ""

            if key:
                # Get response
                component = self.get_openai_response(key, symbol, right_part, search_context, estimation_mode)

                # Handle logical induced negations after the extraction, since the negation is not directly triggered by the text but rather by the constraint structure
                if negated and symbol is not self.empty:
                    old_symbol = symbol
                    neg_symbol = self.negations[symbol]
                    
                    if old_symbol == "==":
                        if 'False' in component:
                            component.replace('False', 'True')
                        elif 'True' in component:
                            component.replace('True', 'False')
                    elif old_symbol in component:
                        component.replace(old_symbol, neg_symbol)

        return component
    
    def get_openai_response(self, key, symbol, right_part, search_context, estimation_mode=True, output_estimate = 10, safety = 1.2):
        """
        Extracts a response from a given text using a predefined template and the OpenAI GPT API. Calculates the resource usage in terms of input and output tokens, including their associated costs.

        :param key: Determines the case.
        :param symbol: Equation symbol.
        :param right_part: Right part of the equation.
        :param search_context: The text from which the constraint component should be extracted.
        :param estimation_mode: Determines if the requests are sent to the OpenAI API (False) or if a plain cost calculation is performed (True), using an estimate for the number of output tokens.
        :param output_estimate: Estimate for the number of output tokens, if mode is 'sim'. 
        :param safety: Safety factor to multiply the total number of tokens with.
        :return: A tuple containing the assistant's response to the second prompt and a dictionary with resource usage details.
        """
        params = self.parameters['openai_params']
        response_2 = ""

        # Construct conversation
        system_prompt = self.parameters['general_prompts']['system_prompt']
        prompt_1 = self.parameters[key]['prompt_1']
        response_1 = self.parameters[key]['response_1']
        prompt_2 = self.parameters['general_prompts']['prompt_2_a'] + " " + symbol + " " + right_part + " " + self.parameters['general_prompts']['prompt_2_b'] + " " + search_context

        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_1},
            {"role": "assistant", "content": response_1},
            {"role": "user", "content": prompt_2}
        ]

        if estimation_mode:
            response_2 = "ESTIMATION MODE"

            # Concatenate all the string objects in messages, using keys and values
            text = ''.join([f"{message['role']}: {message['content']}" for message in messages])
            doc = self.nlp(text)
            # The number of tokens in the doc
            input_tokens = len(doc) * safety
            output_tokens = output_estimate * safety
        
        else:
            client = OpenAI()

            completion = client.chat.completions.create(
                model=params['model'],
                messages=messages
            )
            response_2 = completion.choices[0].message.content

            usage = dict(completion).get('usage')
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens

        # Cost calculation
        input_costs = params['input_price'] * input_tokens / 1000
        output_costs = params['output_price'] * output_tokens / 1000

        self.resources['input_tokens'].append(input_tokens)
        self.resources['output_tokens'].append(output_tokens)
        self.resources['input_costs'].append(input_costs)
        self.resources['output_costs'].append(output_costs)

        return response_2

    def streamline(self, formatted_constraints):
        """
        Streamlines the constraints into the end format relevant for the Gold Standard comparison.

        :param formatted_constraints: An array with the formatted constraints.
        :return: The streamlined constraints in a dict.
        """
        streamlined = {}
        counter = 1
        nr_of_constraints = len(formatted_constraints)

        for i in range(nr_of_constraints):

            step = formatted_constraints[i][0]
            constraint = formatted_constraints[i][1]

            if i < nr_of_constraints - 1:
                succeding_step = formatted_constraints[i + 1][0]
            else:
                succeding_step = "END"
            streamlined[f'c{counter}'] = "({" + step + "}, {" + succeding_step + "}, {directly follows}, {" + constraint + "})"
            counter += 1

        return streamlined

def get_constraint(nlp, builder, text, equality_params, inequality_params, meta_params, enumeration_summary, linebreaks, id=1, verbose=True):
    """
    Conducts a combined search for equality, inequality, and meta (enumeration) constraint items within a given text and formats the constraint according to the builder.

    This function initializes InequalityConstraintSearcher, EqualityConstraintSearcher, and MetaConstraintSearcher, performs the search, and then combines their findings. It also calls the method to format the constraint.

    :param nlp: A spaCy Language model instance used for text processing.
    :param builder: A ConstraintBuilder instance to format the constraint.
    :param text: The text to search within for constraint items.
    :param equality_params: Parameters for the EqualityConstraintSearcher.
    :param inequality_params: Parameters for the InequalityConstraintSearcher.
    :param meta_params: Parameters for the MetaConstraintSearcher.
    :param enumeration_summary: Enumeration summaries for meta constraint searching.
    :param linebreaks: Information on succeding line breaks for meta constraint searching.
    :param id: The ID of the first match found by the current function within the document.
    :param verbose: Parameter to control the printed output. If True, output is printed.
    :return constraint: A dictionary with detailed information about the found constraint. 
    :return id: The ID of the last match + 1 found by the current function within the document.
    """
    first_id = id
    formatted_constraint = [] # Array to fit with the format of the other attributes
    # Initialize searchers
    inequality = InequalityConstraintSearcher(nlp, inequality_params)
    equality = EqualityConstraintSearcher(nlp, equality_params)
    meta = MetaConstraintSearcher(nlp, meta_params)
    
    # Perform the searches and combine match details
    id = inequality.search_constraint_items(text, id)
    id = equality.search_constraint_items(text, id)
    id = meta.search_enum_constraint_items(text, enumeration_summary, id)
    id = meta.search_if_clauses(text, linebreaks, id)
    id = meta.search_for_clauses(text, linebreaks, id)

    # Concatenate findings in output dict
    constraint = {}
    for key in inequality.constraints.keys():
        constraint[key] = inequality.constraints[key] + equality.constraints[key] + meta.constraints[key]

    # If constraints where found, first determine context, then insert 'BOOL' constraints, then rank and connect the constraints
    if len(constraint['id']):
        constraint = meta.determine_context(text, constraint, linebreaks)
        constraint = meta.insert_connections(text, constraint)
        constraint = meta.insert_bool(constraint)
        constraint = meta.sort_and_prune(constraint)

        if len(constraint['id']):
            id = constraint['id'][-1] + 1
            # Build constraint
            formatted_constraint = builder.build_constraint(text, constraint)
        else: # If all constraints removed, reset id
            id = first_id

    if verbose:
        # Unpack negations for visualisation
        combined_negations = [neg[0] for neg in constraint['negation'] if isinstance(neg, tuple)]
        # Highlight text
        ConstraintSearcher(nlp, equality_params).highlight_matches(text, linebreaks, constraint['match'], combined_negations, constraint['type'], constraint['pattern'])

        # Print in tabular format
        combined_table_data = zip(*constraint.values()) 
        # Only print if any of the lists in constraints has elements
        if any(constraint.values()): 
            print()
            print(tabulate(combined_table_data, headers=constraint.keys()))
            print()
            for c in formatted_constraint:
                print(c)

    constraint['formatted'] = formatted_constraint

    return constraint, id

def get_constraints_from_data(nlp, data, equality_params, inequality_params, meta_params, builder_params, verbose=True):
    """
    Conducts a combined search for inequality, equality and meta constraints and builds the constraints for all use cases.

    Wrapper function for get_constraints(). 

    :param nlp: A spaCy Language model instance used for text processing.
    :param data: A dictionary with one dataframe per use case containing pre-processed chunks of text in the 'Lemma' column.  
    :param equality_params: Parameters for the EqualityConstraintSearcher.
    :param inequality_params: Parameters for the InequalityConstraintSearcher.
    :param meta_params: Parameters for the MetaConstraintSearcher.
    :param builder_params: Parameters for the ConstraintBuilder.
    :param verbose: Parameter to control the printed output. If True, output is printed.
    :return constraints: A dictionary with detailed information about the found constraints. 
    """
    builder = ConstraintBuilder(nlp, builder_params, verbose)

    delimiter_line = "+"*80
    constraints_tmp = defaultdict(lambda: defaultdict(list)) # Enable dict assignment before knowing the keys

    for use_case, df in data.items():
        if verbose:
            print()
            print(delimiter_line)
            print(delimiter_line, "\n")
            print(use_case.upper(), "\n")
            print(delimiter_line)
            print(delimiter_line, "\n")

        # Reset the resources
        builder.resources = {'input_tokens':[], 'output_tokens':[], 'input_costs':[], 'output_costs':[]}

        id = 1  # Initialize ID for each use case
        for index, row in df.iterrows(): # Iterate over the rows in the dataframe
            for chunk_index, chunk in enumerate(row['Lemma']):  # Iterate over each chunk in the Lemma column
                if verbose:
                    print()
                    print("++++ CHUNK ++++", "\n")

                # Search for constraints
                new_constraints, id = get_constraint(nlp, builder, chunk, equality_params, inequality_params, meta_params, row['Enumeration'][chunk_index], row['Linebreak'][chunk_index], id, verbose)

                for key, values in new_constraints.items():
                    constraints_tmp[use_case][key].extend(values)
                    if key == 'id':  # Only append to 'index' and 'chunk' when new 'id' is found
                        constraints_tmp[use_case]['index'].extend([index] * len(values))
                        constraints_tmp[use_case]['chunk'].extend([chunk_index] * len(values))

        # Overwrite the successor of the last found constraint item to demark the end, if any constraint was found
        if constraints_tmp[use_case]['successor']:
            constraints_tmp[use_case]['successor'][-1] = (constraints_tmp[use_case]['successor'][-1][0],'END')
        
        # Streamline process steps and IDs of the formatted constraints
        constraints_tmp[use_case]['formatted'] = builder.streamline(constraints_tmp[use_case]['formatted'])
        constraints_tmp[use_case]['resources'] = builder.resources


    # Format to regular dict
    constraints = dict()
    for key, value in constraints_tmp.items():
        constraints[key] = dict(value)

    if verbose:
        print()
        print(delimiter_line)
        print(delimiter_line, "\n")
        print("SUMMARY\n")
        print(delimiter_line)
        print(delimiter_line, "\n")

        total_constraint_items = total_input_costs = total_output_costs = 0
        table_data = []

        for case, c in constraints.items():
            num_components = len(c['id'])
            input_costs = sum(c['resources']['input_costs'])
            output_costs = sum(c['resources']['output_costs'])
            
            # Update totals
            total_constraint_items += num_components
            total_input_costs += input_costs
            total_output_costs += output_costs
            
            # Append data for each case to the table data list
            table_data.append([case, num_components, input_costs, output_costs])

        # Append total row to the table data
        table_data.append(["TOTAL", total_constraint_items, total_input_costs, total_output_costs])

        # Define headers
        headers = ["CASE", "CONSTRAINT COMPONENTS", "COST - INPUT", "COST - OUTPUT"]

        # Print table
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    return constraints