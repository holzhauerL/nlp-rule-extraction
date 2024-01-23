import spacy
from spacy.matcher import Matcher

def gt_mtchs(nlp, text, pattern, pattern_name, verbose=True, context=2):
    """
    Returns the matches of a pattern in a text.

    :param nlp: Pre-loaded SpaCy model.
    :param text: The input text as a string.
    :param pattern: List with dictionaries.
    :param pattern_name: Name to store the pattern in the matcher.
    :param verbose: If True, matches are printed.
    :param context: Number of tokens to include for the context of the match (both sides).
    :return: Set of matches.
    """
    doc = nlp(text)

    # Add the pattern to the matcher
    matcher = Matcher(nlp.vocab)
    matcher.add(pattern_name, [pattern])

    matches = matcher(doc)
    print(matches)
    matches = {"Match": matches, "Span":[],"Context":[]}

    for match_id, start, end in matches["Match"]:
        # Get the matched span
        matches["Span"].append(doc[start:end])
        try:
            matches["Context"].append(doc[start-context:end+context])
        except:
            matches["Context"].append(doc[start:end])
            print("Context out of range")
        if verbose:
            print("Match:",match_id, start, end)
            print("Span:",matches["Span"][-1])
            print("Context:",matches["Context"][-1])

    return matches