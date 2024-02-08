import re
import os
from tabulate import tabulate
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

class Evaluator:
    """
    Class to evaluate the constraints.
    """
    def __init__(self, model, constraints, parameters, gs_file_paths, verbose):
        """
        Initializes the Evaluator.

        :param model: S-BERT model to use for similarity scores. 
        :param constraints: A dictionary with detailed information about the found constraints.
        :param parameters: A dictionary containing various settings and parameters for constraint searching.
        :param gs_file_paths: A dictionary with the paths to the .txt gold standard files.
        :param verbose: Parameter to control the printed output. If True, output is printed.
        """
        self.model = model
        self.constraints = constraints
        self.file_paths = gs_file_paths
        self.parameters = parameters
        self.verbose = verbose

    def _extract_gs(self, file_path):
        """
        Extracts the constraints from the gold standard (GS) .txt file into a dictionary. 

        :param file_path: Path to the GS .txt file. 
        :return: The constraints from the GS in a dictionary.  
        """
        constraints_gs = {}

        with open(file_path, 'r') as file:
            content = file.read()

        for line in content.split('\n'):
            line = line.strip()
            match = re.match(r'^c(\d{1,2}) = (.+)$', line)
            if match:
                current_key = f"c{match.group(1)}"
                constraints_gs[current_key] = match.group(2).replace("   ", " ").replace("  ", " ").replace(" , ", ", ").replace("},{", "}, {")

        return constraints_gs

    def _extract_parts(self, constraint, part):
        """
        Extracts specific parts from a constraint string.
        :param constraint: The constraint string.
        :param part: The part to extract (0 for PROCESS_STEP_1, 1 for PROCESS_STEP_2, 3 for CONSTRAINT).
        :return: Extracted part of the constraint.
        """
        # Trim leading and trailing whitespace
        constraint = constraint.strip()

        # Check and remove the first two and the last two characters
        if constraint.startswith("({") and constraint.endswith("})"):
            constraint = constraint[2:-2]
        else:
            print("WARNING: Constraint",constraint,"format not right. Check start and end characters (should be '({' and '})').")

        # Split the string at the specified delimiters
        parts = constraint.split("}, {")
        if len(parts) < 4:
            print(f"WARNING: Insufficient parts in constraint {constraint}. Expected at least 4 parts.")

        # Return the requested part
        return parts[part]

    def _preprocess_constraint(self, constraint):
        """
        Strips the constraint from all unnecesary characters.

        :param constraint: The constraint string in the format "({PROCESS_STEP_1}, {PROCESS_STEP_2}, {directly follows}, {CONSTRAINT})".
        :return: PROCESS_STEP_1 + " " + PROCESS_STEP_2 + " " + CONSTRAINT.
        """
        return self._extract_parts(constraint, 0) + " " + self._extract_parts(constraint, 1) + " " + self._extract_parts(constraint, 2)
    

    def sbert_similarity_strings(self, text1, text2, underscores=False):
        """
        Determines the pairwise similarity scores of two strings with a pre-trained S-BERT model.

        :param text1: First text.
        :param text2: Second text.
        :param underscores: Determines if underscores should be considered.
        :return: Cosine similarity score.
        """
        if not underscores:
            text1 = text1.replace("_", " ")
            text2 = text2.replace("_", " ")
        
        # Create embeddings
        emb1 = self.model.encode([text1])
        emb2 = self.model.encode([text2])

        # Compute cosine similarity between the embeddings
        return cosine_similarity(emb1, emb2)[0][0]
    
    def compare_references(self, reference):
        """
        Compares each pair of texts in the reference dictionary with and without considering underscores.

        :param reference: Dictionary of text pairs.
        :return: Tabular output of similarity scores.
        """
        if self.verbose:
            results = []
            for key, texts in reference.items():
                score_with_underscores = self.sbert_similarity_strings(texts[0], texts[1], underscores=True)
                score_without_underscores = self.sbert_similarity_strings(texts[0], texts[1], underscores=False)
                results.append([key, score_with_underscores, score_without_underscores])

            headers = ["Reference", "Score w/ underscores", "Score w/o underscores"]
            print(tabulate(results, headers=headers, tablefmt="grid"))

    def sbert_similarty(self, first_set, second_set):
        """
        Determines the pairwise similarity scores of two sets of constraints with a pre-trained S-BERT model.

        Preprocesses the constraints to use only PROCESS_STEP_1 + PROCESS_STEP_2 + CONSTRAINT.
        :param first_set: First set of constraints, expected as a dictionary.
        :param second_set: Second set of constraints, expected as a dictionary.
        :return: Similartiy scores in a dictionary.
        """
        if first_set and second_set:
            # Preprocess and create embeddings for each set of constraints
            embeddings1 = self.model.encode([self._preprocess_constraint(v.replace("_", " ")) for v in first_set.values()], convert_to_tensor=True)
            embeddings2 = self.model.encode([self._preprocess_constraint(v.replace("_", " ")) for v in second_set.values()], convert_to_tensor=True)

            # Compute cosine similarity between the embeddings
            similarity_matrix = cosine_similarity(embeddings1, embeddings2)

            # Create and populate the similarity_scores dictionary
            similarity_scores = {f'{key1} - {key2}': similarity_matrix[i, j] for i, key1 in enumerate(first_set.keys()) for j, key2 in enumerate(second_set.keys())}
        else:
            similarity_scores = {}

        return similarity_scores

    def sbert_similarity_constraints(self, similarity_scores_dict, first_set, second_set, threshold=0):
        """
        Analyzes similarity scores to find matches and organize them into DataFrames.

        :param similarity_scores_dict: Dictionary containing similarity scores between constraints.
        :param first_set: First set of constraints as a dictionary.
        :param second_set: Second set of constraints as a dictionary.
        :param threshold: Minimum similarity score to consider a match. 
        :return: Three DataFrames with matching constraints details.
        """
        # Initialize DataFrames
        cols = ["Constraint pair", "Group similarity", "From", "To", "Similarity"]
        scores_step_1 = pd.DataFrame(columns=cols)
        scores_step_2 = pd.DataFrame(columns=cols)
        scores_constraints = pd.DataFrame(columns=cols)

        # Sort and filter similarity_scores_dict
        sorted_scores = sorted([(k, v) for k, v in similarity_scores_dict.items() if v >= threshold], key=lambda x: x[1], reverse=True)

        # Process each entry in the sorted and filtered list
        processed_keys_set1 = set()
        processed_keys_set2 = set()
        for key_pair, group_similarity in sorted_scores:
            key1, key2 = key_pair.split(' - ')
            
            # Skip if any key has already been processed
            if key1 in processed_keys_set1 or key2 in processed_keys_set2:
                continue

            # Extract parts of the constraints
            from_step_1 = self._extract_parts(first_set[key1], 0)
            to_step_1 = self._extract_parts(second_set[key2], 0)
            from_step_2 = self._extract_parts(first_set[key1], 1)
            to_step_2 = self._extract_parts(second_set[key2], 1)
            from_constraint = self._extract_parts(first_set[key1], 3)
            to_constraint = self._extract_parts(second_set[key2], 3)

            # Calculate S-BERT similarity for each part
            similarity_step_1 = cosine_similarity(self.model.encode([from_step_1]), self.model.encode([to_step_1]))[0][0]
            similarity_step_2 = cosine_similarity(self.model.encode([from_step_2]), self.model.encode([to_step_2]))[0][0]
            similarity_constraint = cosine_similarity(self.model.encode([from_constraint]), self.model.encode([to_constraint]))[0][0]

            # Add to DataFrames
            scores_step_1.loc[len(scores_step_1)] = [key_pair, group_similarity, from_step_1, to_step_1, similarity_step_1]
            scores_step_2.loc[len(scores_step_2)] = [key_pair, group_similarity, from_step_2, to_step_2, similarity_step_2]
            scores_constraints.loc[len(scores_constraints)] = [key_pair, group_similarity, from_constraint, to_constraint, similarity_constraint]

            # Mark keys as processed
            processed_keys_set1.add(key1)
            processed_keys_set2.add(key2)

        return scores_step_1, scores_step_2, scores_constraints
    
    def _precision_recall(self, dfs, cutoff, extracted_set, gs_set, weights):
        """
        Helper function to calculate precision and recall.

        :param dfs: List of DataFrames with the similarity scores for each of the three constraint elements.
        :param cutoff: The threshold value for hard cut. All matches with simalrity scores below this threshold contribute to the precision and recall calculation with a value of 0.
        :param extracted_set: Dictionary of the extracted set of constraints.
        :param gs_set: Dictionary of the gold standard of constraints.
        :param weights: List of individual weights to be used for the evaluation. Must contain three elements which sum up to 1.
        """
        # Only consider scores above the threshold
        for df in dfs:
            df[df < cutoff] = 0
            
        # Count the non-zero values
        non_zero_count = (df != 0).sum().sum()
        
        # Precision and recall with individual weights for PROCESS_STEP_1, PROCESS_STEP_2 and CONSTRAINT.
        rec = sum(non_zero_count / len(gs_set) * w for df, w in zip(dfs, weights))
        if extracted_set:
            prec = sum(non_zero_count / len(extracted_set) * w for df, w in zip(dfs, weights))
        else:
            prec = rec = 0

        return prec, rec

    def evaluate(self, extracted_set, gs_set, scores_step_1, scores_step_2, scores_constraints, weights=[0.2, 0.2, 0.6], cutoff=0.5):
        """
        Evaluates precision and recall for matched constraints.

        :param extracted_set: Dictionary of the extracted set of constraints.
        :param gs_set: Dictionary of the gold standard of constraints.
        :param matches_step_1: DataFrame containing similarity scores for PROCESS_STEP_1.
        :param matches_step_2: DataFrame containing similarity scores for PROCESS_STEP_2.
        :param matches_constraints: DataFrame containing similarity scores for CONSTRAINT.
        :param weights: List of individual weights to be used for the evaluation. Must contain three elements which sum up to 1.
        :param cutoff: The threshold value for hard cut. All matches with simalrity scores below this threshold are not considered for the calculation.
        """
        # Copy DataFrames to prevent modification of original data
        dfs = [scores_step_1["Similarity"].copy(), scores_step_2["Similarity"].copy(), scores_constraints["Similarity"].copy()]

        # Set weights to default if weights does not have the number of required elements or if the sum is not 1
        if not len(weights) == 3 or not sum(weights) == 1:
            print("Default weights used.")
            weights = [1/3, 1/3, 1 - 2/3]

        # Calculate precision and recall with given cutoff and weights
        precision, recall = self._precision_recall(dfs, cutoff, extracted_set, gs_set, weights)   

        # Calculate maximum possible precision and recall (cutoff = 0)
        max_rec = len(dfs[0]) / len(gs_set)
        if extracted_set:
            max_prec = len(dfs[0]) / len(extracted_set)
        else:
            max_prec = max_rec = 0
            print("No constraints extracted.")

        # Calculate precision and recall for each cutoff
        cutoff_linspace = np.arange(0, 1.01, 0.01)  # Including buffer in range
        max_prec_arr = np.full_like(cutoff_linspace, max_prec)
        max_rec_arr = np.full_like(cutoff_linspace, max_rec)

        prec_arr, rec_arr = [], []

        for cut in cutoff_linspace:
            dfs = [scores_step_1["Similarity"].copy(), scores_step_2["Similarity"].copy(), scores_constraints["Similarity"].copy()]

            # Calculate precision and recall for this cutoff step
            prec, rec = self._precision_recall(dfs, cut, extracted_set, gs_set, weights)  

            prec_arr.append(prec)
            rec_arr.append(rec)

        return {'max_prec_arr': max_prec_arr, 'prec_arr': prec_arr, 'max_rec_arr': max_rec_arr, 'rec_arr': rec_arr, 'precision': precision, 'max_prec': max_prec, 'recall': recall, 'max_rec': max_rec}

    def plot_curves(self, results):
        """
        Function to plot the precision and recall curves.

        :param results: Dictionary with the precision and recall scores for all use cases.
        """
        precision = {"id": "prec","label": "Precision"}
        recall = {"id": "rec","label": "Recall"}
        curves = [precision, recall]
        cutoff_linspace = np.arange(0, 1.01, 0.01)  # Including buffer in range

        if self.parameters['save_plot']:
            # Creating folder if not exists and saving the plot
            plot_folder = self.parameters['folder']
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            sub_folder_name = timestamp + "_" + self.parameters['run_id']
            plot_folder = os.path.join(plot_folder, sub_folder_name)
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)

        for curve in curves:
            plt.figure(figsize=self.parameters["plot_figsize"])
            
            for case, scores in results.items():
                c = self.parameters["cmap"][case]
                case_label = case.split(":")[0]
                try: 
                    plt.plot(cutoff_linspace, scores['max_' + curve['id'] + '_arr'], c, ls='--', alpha=self.parameters["plot_alpha"])
                    plt.plot(cutoff_linspace, scores[curve['id'] + '_arr'], c, label=case_label)
                except:
                    pass

            plt.xlabel('Threshold for similarity')
            plt.ylabel(curve['label'])
            plt.title(curve['label'] + ' Curves')
            # plt.title(f'{case}\nWeights: {weights}  Cutoff: {cutoff}')
            plt.legend()
            plt.grid(True)
            plt.xlim([0, 1.05])
            plt.ylim([0, 1.05])

            if self.parameters['save_plot']:
                name = curve['label'].lower()
                plot_filename = f'{name}_{timestamp}.png'
                plot_path = os.path.join(plot_folder, plot_filename)
                plt.savefig(plot_path, bbox_inches='tight')

                # Printing the location of the saved plot
                absolute_plot_path = os.path.abspath(plot_path)
                print(f"Plot saved at: {absolute_plot_path}")

    def evaluate_all(self):
        """
        Wrapper function to evaluate the constraints for all use cases.

        :return: Dictionary with the results of the evaluation.
        """
        results = {}
        for case, path in self.file_paths.items():
            
            extracted = self.constraints[case]['formatted']
            gs = self._extract_gs(path)

            scores = self.sbert_similarty(extracted, gs)

            scores_step_1, scores_step_2, scores_constraints = self.sbert_similarity_constraints(scores, extracted, gs)

            results[case] = self.evaluate(extracted, gs, scores_step_1, scores_step_2, scores_constraints, weights=self.parameters['weights'], cutoff=self.parameters['cutoff'])

        if self.verbose:
             # Preparing data for tabulation
            data = []
            for case, metrics in results.items():
                row = [case, metrics['precision'], metrics['max_prec'], metrics['recall'], metrics['max_rec']]
                data.append(row)

            # Headers for the table
            headers = ["Case", "Precision", "Maximum possible\nprecision", "Recall", "Maximum possible\nrecall"]

            # Generating and printing the table
            table = tabulate(data, headers=headers, tablefmt="grid")
            print(table)

        if self.parameters["plot_curves"]:
            self.plot_curves(results)

        return results
