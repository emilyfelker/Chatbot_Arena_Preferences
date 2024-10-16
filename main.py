import pandas as pd
import zipfile
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import textstat

## TODO:
# do differences and ratios for all combinations
# unbias dataset: with the input triplets of resp_a, resp_b, prompt and mod_a_wins, mod_b_wins, tie, create a
# reversed/flipped dataset and add this to the training set
# try different models
# feature importance plot
# streamline code

def load_data(zip_file_path):
    try:
        # First attempt to load data from the zip file (local environment)
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            with z.open('train.csv') as train_file:
                train_df = pd.read_csv(train_file)
            with z.open('test.csv') as test_file:
                test_df = pd.read_csv(test_file)
        print("Data loaded from zip file.")
    except (FileNotFoundError, zipfile.BadZipFile):
        # If loading from zip fails, assume we're in the Kaggle environment
        print("Zip file not found or invalid. Trying to load from /input folder (Kaggle environment).")
        train_df = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/train.csv')
        test_df = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
        print("Data loaded from /input folder.")
    return train_df, test_df

def add_basic_features(df):
    for col in ['prompt', 'response_a', 'response_b']:
        # Character Count
        df[f'{col}_char_count'] = df[col].str.len()

        # Word List and Word Count
        df[f'{col}_word_list'] = df[col].str.findall(r'\b\w+\b')
        df[f'{col}_word_count'] = df[f'{col}_word_list'].str.len()

        # Sentence Count
        df[f'{col}_sentence_count'] = df[col].str.count(r'[.!?]+')
        df[f'{col}_sentence_count'] = df[f'{col}_sentence_count'].replace(0, 1)  # Avoid division by zero

        # Average Word Length
        df[f'{col}_avg_word_length'] = df[f'{col}_char_count'] / df[f'{col}_word_count'].replace(0, np.nan)

        # Average Sentence Length (in words)
        df[f'{col}_avg_sentence_length'] = df[f'{col}_word_count'] / df[f'{col}_sentence_count']

        # Punctuation Counts
        df[f'{col}_exclamation_count'] = df[col].str.count('!')
        df[f'{col}_question_count'] = df[col].str.count(r'\?')
        df[f'{col}_comma_count'] = df[col].str.count(',')
        df[f'{col}_period_count'] = df[col].str.count(r'\.')
        df[f'{col}_semicolon_count'] = df[col].str.count(';')
        df[f'{col}_colon_count'] = df[col].str.count(':')

        # Personal Pronoun Counts
        df[f'{col}_pronoun_I_count'] = df[col].str.count(r'\bI\b', flags=re.IGNORECASE)
        df[f'{col}_pronoun_you_count'] = df[col].str.count(r'\byou\b', flags=re.IGNORECASE)
        df[f'{col}_pronoun_we_count'] = df[col].str.count(r'\bwe\b', flags=re.IGNORECASE)

        # Type-Token Ratio
        df[f'{col}_type_token_ratio'] = df[f'{col}_word_list'].apply(lambda x: len(set(x)) / max(len(x), 1))

        # Readability Scores using textstat
        df[f'{col}_flesch_reading_ease'] = df[col].apply(lambda text: textstat.flesch_reading_ease(text) if isinstance(text, str) else np.nan)
        df[f'{col}_flesch_kincaid_grade'] = df[col].apply(lambda text: textstat.flesch_kincaid_grade(text) if isinstance(text, str) else np.nan)

    # Differences between responses (Model A - Model B)
    df['char_count_difference'] = df['response_a_char_count'] - df['response_b_char_count']
    df['word_count_difference'] = df['response_a_word_count'] - df['response_b_word_count']
    df['sentence_count_difference'] = df['response_a_sentence_count'] - df['response_b_sentence_count']
    df['avg_word_length_difference'] = df['response_a_avg_word_length'] - df['response_b_avg_word_length']
    df['avg_sentence_length_difference'] = df['response_a_avg_sentence_length'] - df['response_b_avg_sentence_length']
    df['exclamation_count_difference'] = df['response_a_exclamation_count'] - df['response_b_exclamation_count']
    df['question_count_difference'] = df['response_a_question_count'] - df['response_b_question_count']
    df['pronoun_I_count_difference'] = df['response_a_pronoun_I_count'] - df['response_b_pronoun_I_count']
    df['pronoun_you_count_difference'] = df['response_a_pronoun_you_count'] - df['response_b_pronoun_you_count']
    df['type_token_ratio_difference'] = df['response_a_type_token_ratio'] - df['response_b_type_token_ratio']
    df['flesch_reading_ease_difference'] = df['response_a_flesch_reading_ease'] - df['response_b_flesch_reading_ease']
    df['flesch_kincaid_grade_difference'] = df['response_a_flesch_kincaid_grade'] - df['response_b_flesch_kincaid_grade']

    # Ratios involving prompt and responses
    df['response_a_char_count_to_prompt_char_count_ratio'] = df['response_a_char_count'] / df['prompt_char_count'].replace(0, np.nan)
    df['response_b_char_count_to_prompt_char_count_ratio'] = df['response_b_char_count'] / df['prompt_char_count'].replace(0, np.nan)

    # Ratios between responses (Model A to Model B)
    df['response_a_char_count_to_response_b_char_count_ratio'] = df['response_a_char_count'] / df['response_b_char_count'].replace(0, np.nan)
    df['response_a_word_count_to_response_b_word_count_ratio'] = df['response_a_word_count'] / df['response_b_word_count'].replace(0, np.nan)
    df['response_a_sentence_count_to_response_b_sentence_count_ratio'] = df['response_a_sentence_count'] / df['response_b_sentence_count'].replace(0, np.nan)

    # Clean up temporary columns
    df.drop(columns=[col for col in df.columns if '_word_list' in col], inplace=True)

    return df

def prepare_data(train_df):
    # Select features
    feature_cols = [col for col in train_df.columns if any(keyword in col for keyword in [
        '_char_count', '_word_count', '_sentence_count',
        '_avg_word_length', '_avg_sentence_length',
        '_exclamation_count', '_question_count', '_comma_count', '_period_count',
        '_semicolon_count', '_colon_count', '_pronoun_I_count', '_pronoun_you_count',
        '_pronoun_we_count', '_type_token_ratio', '_flesch_reading_ease', '_flesch_kincaid_grade',
        '_difference', '_ratio'
    ])]

    # Multi-class target: 0 = Model A wins, 1 = Model B wins, 2 = Tie
    def get_target(row):
        if row['winner_model_a'] == 1:
            return 0  # Model A wins
        elif row['winner_model_b'] == 1:
            return 1  # Model B wins
        else:
            return 2  # Tie

    train_df['target'] = train_df.apply(get_target, axis=1)

    X = train_df[feature_cols]
    y = train_df['target']

    # Handle any NaN values
    X = X.fillna(0)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_val, y_train, y_val, scaler


def train_model(X_train, y_train):
    # Use LogisticRegression with multi-class support
    model = LogisticRegression(solver='lbfgs', max_iter=2000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    # Predict probabilities for the validation set
    y_val_pred_proba = model.predict_proba(X_val)

    # Compute the multi-class log loss
    loss = log_loss(y_val, y_val_pred_proba)
    print(f'Validation Log Loss: {loss}')

    # Predict actual class labels
    y_val_pred = model.predict(X_val)

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)
    plot_confusion_matrix(cm)


def plot_confusion_matrix(cm, filename='confusion_matrix.png'):
    # Rearrange the confusion matrix to have "Tie" in the middle
    # Original order: [Model A Wins, Model B Wins, Tie]
    # New order: [Model A Wins, Tie, Model B Wins]
    reordered_indices = [0, 2, 1]
    reordered_cm = cm[reordered_indices][:, reordered_indices]

    # Normalize the confusion matrix by dividing each row by the sum of that row
    cm_relative = reordered_cm.astype('float') / reordered_cm.sum(axis=1)[:, np.newaxis]

    # Labels for the confusion matrix
    labels = ['Model A Wins', 'Tie', 'Model B Wins']

    # Create heatmap for confusion matrix with relative counts
    sns.heatmap(cm_relative, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Relative Counts)')

    # Save the plot to a file
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Clear the plot to avoid overlapping plots in future calls
    plt.clf()


def make_predictions(model, test_df, scaler):
    # Add features to the test set
    test_df = add_basic_features(test_df)

    # Select the same features used for training
    feature_cols = [col for col in test_df.columns if any(keyword in col for keyword in [
        '_char_count', '_word_count', '_sentence_count',
        '_avg_word_length', '_avg_sentence_length',
        '_exclamation_count', '_question_count', '_comma_count', '_period_count',
        '_semicolon_count', '_colon_count', '_pronoun_I_count', '_pronoun_you_count',
        '_pronoun_we_count', '_type_token_ratio', '_flesch_reading_ease', '_flesch_kincaid_grade',
        '_difference', '_ratio'
    ])]

    # Handle any NaN values
    X_test = test_df[feature_cols].fillna(0)

    # Scale the test data using the same scaler
    X_test_scaled = scaler.transform(X_test)

    # Predict probabilities for the test set
    y_test_pred_proba = model.predict_proba(X_test_scaled)

    # Create a submission dataframe
    submission_df = test_df[['id']].copy()
    submission_df['winner_model_a'] = y_test_pred_proba[:, 0]  # Probability of Model A winning
    submission_df['winner_model_b'] = y_test_pred_proba[:, 1]  # Probability of Model B winning
    submission_df['winner_model_tie'] = y_test_pred_proba[:, 2]  # Probability of a tie

    return submission_df


def create_submission_file(submission_df, filename='submission.csv'):
    submission_df.to_csv(filename, index=False)
    print(f'Submission file saved as {filename}')


def main():
    # Load dataset
    train_df, test_df = load_data('data/lmsys-chatbot-arena.zip')

    # Add features
    train_df = add_basic_features(train_df)
    test_df = add_basic_features(test_df)

    # Prepare data for training
    X_train, X_val, y_train, y_val, scaler = prepare_data(train_df)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_val, y_val)

    # Prepare test data
    submission = make_predictions(model, test_df, scaler)

    # Create and save the submission file
    create_submission_file(submission)


if __name__ == '__main__':
    main()