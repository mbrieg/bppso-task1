import pandas as pd
import matplotlib.pyplot as plt
# import pm4py as pm
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score


class Helper:
    """
    A utility class containing methods to assist with the analysis, modeling,
    and evaluation of event log data for the Business Process Simulation
    and Optimization exercise.

    This class provides tools for data overview, process timeline analysis,
    Petri net metric calculation, and machine learning model evaluation.
    """

    def show_overview_table(self, df, sample_size=3):
        """
        Generates a summary DataFrame for initial data exploration (counts, types, nulls, samples).

        Args:
            df (pd.DataFrame): The input event log DataFrame.
            sample_size (int, optional): The maximum number of sample unique values
                to display per column. Defaults to 3.

        Returns:
            pd.DataFrame: A table summarizing the key characteristics of the DataFrame,
                resetting the index and renaming the index column to 'Activity'.
        """
        df_overview = pd.DataFrame({
            'Unique Values': df.nunique(),
            'Data Type': df.apply(lambda col: col.dtype),
            'Elements not null': df.apply(lambda col: col.notna().sum()),
            'Sample values': df.apply(lambda col: list(col.unique()[:sample_size]))
        })
        pd.set_option('display.max_colwidth', 1000)
        return df_overview.reset_index().rename(columns={'index': 'Activity'})

    def show_timeline_table(self, df):
        """
        Calculates and displays the start and end dates for key events ('A_Create Application'
        and 'O_Create Offer').

        Args:
            df (pd.DataFrame): The input event log DataFrame, which must contain
                'concept:name' and 'time:timestamp' columns.

        Returns:
            pd.DataFrame: A table showing the minimum (start) and maximum (end)
                timestamps for the 'Application' and 'Offer' events.
        """
        start_app = df.loc[df['concept:name'] == 'A_Create Application', 'time:timestamp'].min()
        end_app = df.loc[df['concept:name'] == 'A_Create Application', 'time:timestamp'].max()
        start_off = df.loc[df['concept:name'] == 'O_Create Offer', 'time:timestamp'].min()
        end_off = df.loc[df['concept:name'] == 'O_Create Offer', 'time:timestamp'].max()
        timeline_tbl = pd.DataFrame({
            'Event Origin': ['Application', 'Offer'],
            'Start date': [start_app, start_off],
            'End date': [end_app, end_off]
        })
        return timeline_tbl

    def calculate_frequency_table(self, df, column):
        """
        Calculates the count, percentage, and cumulative percentage for unique values in a column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column (str): The name of the column to analyze (e.g., 'concept:name'
                for activity frequency).

        Returns:
            pd.DataFrame: A frequency table with 'Count', 'Percentage', and
                'Cumulative' columns.
        """
        count = df[column].value_counts()
        percentages = df[column].value_counts(normalize=True) * 100
        freq_tbl = pd.DataFrame({
            'Count': count,
            'Percentage': percentages.map('{:.3f}%'.format),
            'Cumulative': percentages.cumsum().map('{:.3f}%'.format)
        })
        freq_tbl.index.name = 'Name'
        freq_tbl = freq_tbl.reset_index()
        return freq_tbl

    def calculate_transitions_table(self, net):
        """
        Calculates statistics regarding the transitions in a Petri net, focusing
        on the count and percentage of silent transitions.

        Args:
            net (pm4py.objects.petri.net.PetriNet): The discovered Petri net object.

        Returns:
            pd.DataFrame: A one-row table with 'Total transitions', 'Silent transitions',
                and their 'Percentage'.
        """
        transitions = net.transitions
        silent_transitions = [t for t in transitions if getattr(t, "label", None) is None]
        percentage = len(silent_transitions) / len(transitions) * 100
        transitions_stats = pd.DataFrame({
            'Total transitions': [len(transitions)],
            'Silent transitions': [len(silent_transitions)],
            'Percentage': [f"{percentage:.3f}%"]
        })
        return transitions_stats

    def calculate_simplicity_metrics(self, net, model):
        """
        Calculates structural simplicity metrics for a discovered Petri net model.

        Metrics include node count, arc count, gateway counts, Density,
        Coefficient of Connectivity (CNC), and Sequentiality.

        Args:
            net (pm4py.objects.petri.net.PetriNet): The discovered Petri net object.
            model (pm4py.objects.petri.net.PetriNet): The model object (often
                the same as net, used to retrieve nodes).

        Returns:
            pd.DataFrame: A one-row table containing the computed simplicity metrics.
        """
        nodes = model.get_nodes()
        arcs = net.arcs
        in_arcs = []
        out_arcs = []
        seq_arc_count = 0
        split_gateways = 0
        join_gateways = 0

        for p in net.places:
            in_arcs.append(p.in_arcs)
            out_arcs.append(p.out_arcs)
            if len(p.out_arcs) > 1:     # node is split gateway
                split_gateways += 1
                seq_arc_count += len(p.out_arcs)    # arcs that are not just connectors
            elif len(p.in_arcs) > 1:    # node is join gateway
                join_gateways += 1
                seq_arc_count += len(p.in_arcs)
        gateway_count = split_gateways + join_gateways

        metrics = pd.DataFrame({
            'Nodes': [len(nodes)],
            'Arcs': [len(arcs)],
            'In arcs': [len(in_arcs)],
            'Out arcs': [len(out_arcs)],
            'Gateways': [gateway_count],
            'Density': [len(arcs) / len(nodes) * len(nodes)],
            'Coefficient of Connectivity (CNC)': [len(arcs) / len(nodes)],
            'Sequentiality': [seq_arc_count / len(arcs)]
        })
        return metrics

    def combine_simplicity_dataframes(self, df1, df2):
        """
        Combines two simplicity metrics DataFrames and ensures a clean, single-row output.

        The concatenation is followed by forward-fill and back-fill operations
        to ensure all columns in the final single-row DataFrame have values,
        handling potential NaN values created during concatenation.

        Args:
            df1 (pd.DataFrame): The first input DataFrame (e.g., simple metrics).
            df2 (pd.DataFrame): The second input DataFrame (e.g., transition metrics).

        Returns:
            pd.DataFrame: A single-row DataFrame containing the combined metrics.
        """
        temp = pd.concat([df1, df2], ignore_index=True)
        return temp.ffill().bfill().head(1)

    def train_model(self, X, y):
        """
        Prepares data, trains a Logistic Regression model, and returns the
        resulting Confusion Matrix.

        The data is split into 80/20 train/test sets, scaled using StandardScaler,
        and the Logistic Regression is trained with balanced class weights.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series or np.array): Target labels.

        Returns:
            np.array: A 2x2 confusion matrix (CM) from the test set predictions.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=True, stratify=y)
        log_reg = LogisticRegression(random_state=True, max_iter=1000, class_weight='balanced')
        sc = StandardScaler()  # E[X] = 0, variance 1
        X_train_scaled = sc.fit_transform(X_train)
        X_test_scaled = sc.transform(X_test)

        # Fit logistic regression on training data
        log_reg.fit(X_train_scaled, y_train)

        y_test_pred = log_reg.predict(X_test_scaled)

        y_test_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

        return [y_test, y_test_pred, y_test_proba]

    def calculate_reg_metrics(self, cm):
        """
        Calculates key performance metrics from a confusion matrix (CM).

        Metrics include Sensitivity, Specificity, Accuracy, FNR, FPR, and Precision.

        Args:
            cm (np.array): A 2x2 confusion matrix.

        Returns:
            pd.DataFrame: A one-row table with all calculated regression metrics.
        """
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]

        sensitivity = TP / (TP + FN)
        accuracy = (TP + TN) / (TN + FP + FN + TP)
        specificity = TN / (TN + FP)

        return pd.DataFrame({
            'Sensitivity': [sensitivity],
            'Specificity': [specificity],
            'FNR': [1 - sensitivity],
            'FPR': [1 - specificity],
            'Accuracy': [accuracy],
            'Mis classification': [1 - accuracy],
            'Precision': [TP / (TP + FP)]
        })

    def plot_reg_results(self, y_test, y_test_proba, name):
        """
        Generates and displays the Receiver Operating Characteristic (ROC) curve
        and calculates the Area Under the Curve (AUC) for both training and
        test datasets.

        Note: This method assumes that 'model', 'X_train_scaled', 'X_test_scaled',
              'y_train', and 'y_test' are globally available or defined within
              the scope where this method is called (e.g., the notebook).
        """
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_proba)   # ROC curve
        auc_test = roc_auc_score(y_test, y_test_proba)  # AUC-ROC

        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr_test, tpr_test, label=f'Test (AUC = {auc_test:.2f})', linewidth=2, color='red')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)', linewidth=1.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Logistic Regression Model\n(' + name + ')',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        plt.tight_layout()
        plt.show()
