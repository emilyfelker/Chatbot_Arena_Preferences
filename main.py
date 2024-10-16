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
# streamline code (incl. print statements and comments)
# check for improving computational efficiency
# unbias dataset
# try different models
# feature importance plot
# update ipython notebook for kaggle

def load_data(zip_file_path):
    print("Loading data...")
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
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df

def calculate_features(df):
    inputs = ['prompt', 'response_a', 'response_b']
    print("Calculating basic features for each input...")

    feature_dict = {}  # Dictionary to hold all new features

    for col in inputs:
        print(f"Calculating features for {col}...")

        # Character Count
        feature_dict[f'{col}_char_count'] = df[col].str.len()

        # Word List and Word Count
        word_list = df[col].str.findall(r'\b\w+\b')
        feature_dict[f'{col}_word_list'] = word_list
        feature_dict[f'{col}_word_count'] = word_list.str.len()

        # Sentence Count
        sentence_count = df[col].str.count(r'[.!?]+')
        sentence_count = sentence_count.replace(0, 1)  # Avoid division by zero
        feature_dict[f'{col}_sentence_count'] = sentence_count

        # Average Word Length
        feature_dict[f'{col}_avg_word_length'] = feature_dict[f'{col}_char_count'] / feature_dict[f'{col}_word_count'].replace(0, np.nan)

        # Average Sentence Length (in words)
        feature_dict[f'{col}_avg_sentence_length'] = feature_dict[f'{col}_word_count'] / sentence_count

        # Punctuation Counts
        feature_dict[f'{col}_exclamation_count'] = df[col].str.count('!')
        feature_dict[f'{col}_question_count'] = df[col].str.count(r'\?')
        feature_dict[f'{col}_comma_count'] = df[col].str.count(',')
        feature_dict[f'{col}_period_count'] = df[col].str.count(r'\.')
        feature_dict[f'{col}_semicolon_count'] = df[col].str.count(';')
        feature_dict[f'{col}_colon_count'] = df[col].str.count(':')

        # Personal Pronoun Counts
        feature_dict[f'{col}_pronoun_I_count'] = df[col].str.count(r'\bI\b', flags=re.IGNORECASE)
        feature_dict[f'{col}_pronoun_you_count'] = df[col].str.count(r'\byou\b', flags=re.IGNORECASE)
        feature_dict[f'{col}_pronoun_we_count'] = df[col].str.count(r'\bwe\b', flags=re.IGNORECASE)

        # Type-Token Ratio
        feature_dict[f'{col}_type_token_ratio'] = word_list.apply(lambda x: len(set(x)) / max(len(x), 1))

        # Readability Scores using textstat
        feature_dict[f'{col}_flesch_reading_ease'] = df[col].apply(lambda text: textstat.flesch_reading_ease(text) if isinstance(text, str) else np.nan)
        feature_dict[f'{col}_flesch_kincaid_grade'] = df[col].apply(lambda text: textstat.flesch_kincaid_grade(text) if isinstance(text, str) else np.nan)

    # Convert the feature dictionary to a DataFrame and concatenate
    feature_df = pd.DataFrame(feature_dict)
    df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

    # Remove temporary columns like word lists
    print([col for col in df.columns if '_word_list' in col])
    df.drop(columns=[col for col in df.columns if '_word_list' in col], inplace=True)

    print("Finished calculating basic features.")
    return df

def calculate_differences_and_ratios(df):
    print("Calculating differences and ratios between inputs...")

    pairs = [('prompt', 'response_a'), ('prompt', 'response_b'), ('response_a', 'response_b')]

    basic_features = ['char_count', 'word_count', 'sentence_count', 'avg_word_length', 'avg_sentence_length',
                      'exclamation_count', 'question_count', 'comma_count', 'period_count', 'semicolon_count',
                      'colon_count', 'pronoun_I_count', 'pronoun_you_count', 'pronoun_we_count', 'type_token_ratio',
                      'flesch_reading_ease', 'flesch_kincaid_grade']

    diff_ratio_dict = {}  # Dictionary to hold difference and ratio features

    for feature in basic_features:
        for col1, col2 in pairs:
            diff = df[f'{col1}_{feature}'] - df[f'{col2}_{feature}']
            ratio = df[f'{col1}_{feature}'] / df[f'{col2}_{feature}'].replace(0, np.nan)

            # Add to the dictionary with full arrays
            diff_ratio_dict[f'{col1}_{col2}_{feature}_difference'] = diff
            diff_ratio_dict[f'{col1}_{col2}_{feature}_ratio'] = ratio

    # Convert the difference and ratio dictionary to a DataFrame
    diff_ratio_df = pd.DataFrame(diff_ratio_dict)

    # Ensure the index is passed explicitly to avoid scalar errors
    diff_ratio_df.index = df.index

    # Concatenate the differences and ratios DataFrame to the original DataFrame
    df = pd.concat([df.reset_index(drop=True), diff_ratio_df.reset_index(drop=True)], axis=1)

    print("Finished calculating differences and ratios.")
    return df

def add_basic_features(df):
    print("Adding basic features...")
    df = calculate_features(df)
    df = calculate_differences_and_ratios(df)
    print("Finished adding features.")
    return df

def prepare_data(train_df):
    print("Preparing data for training...")
    feature_cols = [col for col in train_df.columns if any(keyword in col for keyword in [
        '_char_count', '_word_count', '_sentence_count',
        '_avg_word_length', '_avg_sentence_length',
        '_exclamation_count', '_question_count', '_comma_count', '_period_count',
        '_semicolon_count', '_colon_count', '_pronoun_I_count', '_pronoun_you_count',
        '_pronoun_we_count', '_type_token_ratio', '_flesch_reading_ease', '_flesch_kincaid_grade',
        '_difference', '_ratio'
    ])]

    train_df['target'] = train_df.apply(lambda row: 0 if row['winner_model_a'] == 1 else (1 if row['winner_model_b'] == 1 else 2), axis=1)
    X = train_df[feature_cols]
    y = train_df['target']
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data prepared: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples.")
    return X_train, X_val, y_train, y_val, scaler

def train_model(X_train, y_train):
    print("Training model...")
    model = LogisticRegression(solver='lbfgs', max_iter=2000)
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

def evaluate_model(model, X_val, y_val):
    print("Evaluating model...")
    y_val_pred_proba = model.predict_proba(X_val)
    loss = log_loss(y_val, y_val_pred_proba)
    print(f'Validation Log Loss: {loss}')
    y_val_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_val_pred)
    plot_confusion_matrix(cm)
    print("Model evaluation complete.")

def plot_confusion_matrix(cm, filename='confusion_matrix.png'):
    reordered_indices = [0, 2, 1]
    reordered_cm = cm[reordered_indices][:, reordered_indices]
    cm_relative = reordered_cm.astype('float') / reordered_cm.sum(axis=1)[:, np.newaxis]
    labels = ['Model A Wins', 'Tie', 'Model B Wins']
    sns.heatmap(cm_relative, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Relative Counts)')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.clf()
    print(f"Confusion matrix saved as {filename}.")

def make_predictions(model, test_df, scaler):
    print("Making predictions on the test set...")
    feature_cols = [col for col in test_df.columns if any(keyword in col for keyword in [
        '_char_count', '_word_count', '_sentence_count',
        '_avg_word_length', '_avg_sentence_length',
        '_exclamation_count', '_question_count', '_comma_count', '_period_count',
        '_semicolon_count', '_colon_count', '_pronoun_I_count', '_pronoun_you_count',
        '_pronoun_we_count', '_type_token_ratio', '_flesch_reading_ease', '_flesch_kincaid_grade',
        '_difference', '_ratio'
    ])]
    X_test = test_df[feature_cols].fillna(0)
    X_test_scaled = scaler.transform(X_test)
    y_test_pred_proba = model.predict_proba(X_test_scaled)
    submission_df = test_df[['id']].copy()
    submission_df['winner_model_a'] = y_test_pred_proba[:, 0]
    submission_df['winner_model_b'] = y_test_pred_proba[:, 1]
    submission_df['winner_model_tie'] = y_test_pred_proba[:, 2]
    print("Test set predictions complete.")
    return submission_df

def create_submission_file(submission_df, filename='submission.csv'):
    submission_df.to_csv(filename, index=False)
    print(f"Submission file saved as {filename}")

def main():
    print("Starting main program...")
    train_df, test_df = load_data('data/lmsys-chatbot-arena.zip')
    print("Adding features to the train and test data...")
    train_df = add_basic_features(train_df)
    test_df = add_basic_features(test_df)
    X_train, X_val, y_train, y_val, scaler = prepare_data(train_df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_val, y_val)
    submission = make_predictions(model, test_df, scaler)
    create_submission_file(submission)

if __name__ == '__main__':
    main()