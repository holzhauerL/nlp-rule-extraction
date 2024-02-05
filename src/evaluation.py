import re
import os
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
        self.results = {}

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

    def sbert_similarty(self, first_set, second_set):
        """
        Determines the pairwise similarity scores of two sets of constraints with a pre-trained S-BERT model.
        Preprocesses the constraints to use only PROCESS_STEP_1 + PROCESS_STEP_2 + CONSTRAINT.
        :param first_set: First set of constraints, expected as a dictionary.
        :param second_set: Second set of constraints, expected as a dictionary.
        :return: Similartiy scores in a dictionary.
        """
        
        # Preprocess and create embeddings for each set of constraints
        embeddings1 = self.model.encode([self._preprocess_constraint(v.replace("_", " ")) for v in first_set.values()], convert_to_tensor=True)
        embeddings2 = self.model.encode([self._preprocess_constraint(v.replace("_", " ")) for v in second_set.values()], convert_to_tensor=True)

        # Compute cosine similarity between the embeddings
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)

        # Create a dictionary to store similarity scores
        similarity_scores = {}

        # Populate the similarity_scores dictionary
        for i, key1 in enumerate(first_set.keys()):
            for j, key2 in enumerate(second_set.keys()):
                similarity_scores[f'{key1} - {key2}'] = similarity_matrix[i, j]

        return similarity_scores

    def sbert_similarity_constraints(self, similarity_scores_dict, first_set, second_set, threshold=0.5, mode='unique'):
        """
        Analyzes similarity scores to find matches and organize them into DataFrames.
        :param similarity_scores_dict: Dictionary containing similarity scores between constraints.
        :param first_set: First set of constraints as a dictionary.
        :param second_set: Second set of constraints as a dictionary.
        :param threshold: Minimum similarity score to consider a match. Default is 0.5.
        :param mode: Mode of matching, either 'unique' or 'multiple'. Default is 'unique'.
        :return: Three DataFrames with matching constraints details.
        """
        # Initialize DataFrames
        cols = ["Constraint pair", "Group similarity", "From", "To", "Similarity"]
        matches_step_1 = pd.DataFrame(columns=cols)
        matches_step_2 = pd.DataFrame(columns=cols)
        matches_constraints = pd.DataFrame(columns=cols)

        # Sort and filter similarity_scores_dict
        sorted_scores = sorted([(k, v) for k, v in similarity_scores_dict.items() if v >= threshold], key=lambda x: x[1], reverse=True)

        # Process each entry in the sorted and filtered list
        processed_keys_set1 = set()
        processed_keys_set2 = set()
        for key_pair, group_similarity in sorted_scores:
            key1, key2 = key_pair.split(' - ')
            
            # In 'unique' mode, skip if any key has already been processed
            if mode == 'unique':
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
            matches_step_1.loc[len(matches_step_1)] = [key_pair, group_similarity, from_step_1, to_step_1, similarity_step_1]
            matches_step_2.loc[len(matches_step_2)] = [key_pair, group_similarity, from_step_2, to_step_2, similarity_step_2]
            matches_constraints.loc[len(matches_constraints)] = [key_pair, group_similarity, from_constraint, to_constraint, similarity_constraint]

            # Mark keys as processed
            processed_keys_set1.add(key1)
            processed_keys_set2.add(key2)

        return matches_step_1, matches_step_2, matches_constraints

    def evaluate(self, extracted_set, gs_set, matches_step_1, matches_step_2, matches_constraints, individual_weights=True, weights=[0.2, 0.2, 0.6], hard_cut=True, threshold=0.8, plot_curves=True, save_plot=True):
        """
        Evaluates precision and recall for matched constraints and optionally plots precision-recall curves.

        :param extracted_set: Dictionary of the extracted set of constraints.
        :param gs_set: Dictionary of the gold standard of constraints.
        :param matches_step_1: DataFrame containing similarity scores for PROCESS_STEP_1.
        :param matches_step_2: DataFrame containing similarity scores for PROCESS_STEP_2.
        :param matches_constraints: DataFrame containing similarity scores for CONSTRAINT.
        :param individual_weights: Flag to use individual weights for evaluation. Default is True.
        :param weights: List of individual weights to be used for the evaluation. Only considered if individual_weights is True. Default is [0.2, 0.2, 0.6].
        :param hard_cut: Flag to apply a hard cut based on the similarity threshold. Default is True.
        :param threshold: The threshold value for hard cut. Only considered if hard_cut is True. All matches with simalrity scores below this threshold contribute to the precision and recall calculation with a value of 0. threshold == 0 yields the same result as hard_cut == False. Default for threshold is 0.8. 
        :param plot_curves: Flag to generate precision-recall curves plot. Default is True.
        :param save_plot: Flag to save the generated precision-recall curves plot. Default is True.
        """
        # Copy DataFrames to prevent modification of original data
        dfs = [matches_step_1["Similarity"].copy(), matches_step_2["Similarity"].copy(), matches_constraints["Similarity"].copy()]

        # Apply hard cut if enabled
        if hard_cut:
            for df in dfs:
                df[df < threshold] = 0

        # Set weights to default if individual_weights is not set
        if not individual_weights:
            weights = [1/3, 1/3, 1 - 2/3]

        # Calculate precision and recall
        prec_step_1 = dfs[0].sum() / len(extracted_set)
        prec_step_2 = dfs[1].sum() / len(extracted_set)
        prec_constraints = dfs[2].sum() / len(extracted_set)
        max_prec = len(dfs[0]) / len(extracted_set)

        precision = weights[0] * prec_step_1 + weights[1] * prec_step_2 + weights[2] * prec_constraints

        rec_step_1 = dfs[0].sum() / len(gs_set)
        rec_step_2 = dfs[1].sum() / len(gs_set)
        rec_constraints = dfs[2].sum() / len(gs_set)
        max_rec = len(dfs[0]) / len(gs_set)

        recall = weights[0] * rec_step_1 + weights[1] * rec_step_2 + weights[2] * rec_constraints

        if self.verbose:
            print("Precision:", precision)
            print("Maximum precision:", max_prec)
            print("Recall:", recall)
            print("Maximum recall:", max_rec)

        # Generate and save the plot if enabled
        if plot_curves:
            thresholds = np.arange(0, 1.01, 0.01)  # Including buffer in range
            max_prec_arr = np.full_like(thresholds, max_prec)
            max_rec_arr = np.full_like(thresholds, max_rec)

            prec_arr, rec_arr = [], []
            prec_wgthd_arr, rec_wgthd_arr = [], []

            # Calculate precision and recall for each threshold
            for th in thresholds:
                dfs = [matches_step_1["Similarity"].copy(), matches_step_2["Similarity"].copy(), matches_constraints["Similarity"].copy()]
                for df in dfs:
                    df[df < th] = 0
                
                # Precision and recall with default weights
                prec_default = sum(df.sum() / len(extracted_set) for df in dfs) / 3
                rec_default = sum(df.sum() / len(gs_set) for df in dfs) / 3
                
                # Precision and recall with individual weights
                prec_wgthd = sum(df.sum() / len(extracted_set) * w for df, w in zip(dfs, weights))
                rec_wgthd = sum(df.sum() / len(gs_set) * w for df, w in zip(dfs, weights))

                prec_arr.append(prec_default)
                rec_arr.append(rec_default)
                prec_wgthd_arr.append(prec_wgthd)
                rec_wgthd_arr.append(rec_wgthd)

            # Plotting 
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, max_prec_arr, 'r--', label='Maximum precision')
            plt.plot(thresholds, max_rec_arr, 'b--', label='Maximum recall')
            plt.plot(thresholds, prec_wgthd_arr, 'r', alpha=0.8, label='Precision (individual weights)')
            plt.plot(thresholds, rec_wgthd_arr, 'b', alpha=0.8, label='Recall (individual weights)')
            plt.plot(thresholds, prec_arr, 'r', alpha=0.4, label='Precision (default weights)')
            plt.plot(thresholds, rec_arr, 'b', alpha=0.4, label='Recall (default weights)')

            plt.xlabel('Threshold for similarity')
            plt.ylabel('Evaluation metrics')
            plt.title('Precision and Recall Curves')
            plt.legend()
            plt.grid(True)
            plt.xlim([0, 1.05])
            plt.ylim([0, 1.05])
            plt.text(1.1, 0.4, f"# of extracted constraints: {len(extracted_set)}\n# of constraints in the Gold Standard: {len(gs_set)}\n\nWeights are attributed to PROCESS_STEP_1,\nPROCESS_STEP_2 and the CONSTRAINT of each\nconstraint item. \n\nIndividual weights: {weights}\nDefault weights: [0.33, 0.33, 0.34]", horizontalalignment='left')

            if save_plot:
                # Creating folder if not exists and saving the plot
                plot_folder = 'plots'
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plot_filename = f'precision_recall_curves_{timestamp}.png'
                plot_path = os.path.join(plot_folder, plot_filename)
                plt.savefig(plot_path, bbox_inches='tight')

                # Printing the location of the saved plot
                absolute_plot_path = os.path.abspath(plot_path)
                print(f"Plot saved at: {absolute_plot_path}")

    def evaluate_all(self):
        
        for case, path in self.file_paths.items():

            print()
            print(case.upper())
            print()
            
            extracted = self.constraints[case]["formatted"]
            gs = self._extract_gs(path)

            scores = self.sbert_similarty(extracted, gs)

            scores_step_1, scores_step_2, scores_constraints = self.sbert_similarity_constraints(scores, extracted, gs, threshold=0, mode='unique')

            self.evaluate(extracted, gs, scores_step_1, scores_step_2, scores_constraints, individual_weights=True, weights=[0.2, 0.2, 0.6], hard_cut=False, threshold=0.8, plot_curves=True, save_plot=False)
