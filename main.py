import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## TODO:
# add a ton of features
# unbias dataset

def load_data(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open('train.csv') as train_file:
            train_df = pd.read_csv(train_file)
        with z.open('test.csv') as test_file:
            test_df = pd.read_csv(test_file)
    return train_df, test_df


def add_basic_features(df):
    df['response_a_length'] = df['response_a'].apply(len)
    df['response_b_length'] = df['response_b'].apply(len)
    df['length_difference'] = df['response_a_length'] - df['response_b_length']
    return df


def prepare_data(train_df):
    features = ['response_a_length', 'response_b_length', 'length_difference']

    # Multi-class target: 0 = Model A wins, 1 = Model B wins, 2 = Tie
    def get_target(row):
        if row['winner_model_a'] == 1:
            return 0  # Model A wins
        elif row['winner_model_b'] == 1:
            return 1  # Model B wins
        else:
            return 2  # Tie

    train_df['target'] = train_df.apply(get_target, axis=1)

    X = train_df[features]
    y = train_df['target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_val, y_train, y_val


def train_model(X_train, y_train):
    # Use LogisticRegression with multi-class support (using softmax under the hood)
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    # Predict probabilities for each class for the validation set
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
    # Rearrange the confusion matrix to have "tie" in the middle
    # Original order: [Model A Wins, Model B Wins, Tie]
    # New order: [Model A Wins, Tie, Model B Wins]
    reordered_cm = cm[[0, 2, 1]][:, [0, 2, 1]]  # Reorder rows and columns

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


def make_predictions(model, test_df):
    # Add the same features to the test set
    test_df = add_basic_features(test_df)

    # Extract features for prediction
    features = ['response_a_length', 'response_b_length', 'length_difference']
    X_test = test_df[features]

    # Predict probabilities for the test set
    y_test_pred_proba = model.predict_proba(X_test)

    # Create a submission dataframe
    submission_df = test_df[['id']].copy()
    submission_df['winner_model_a'] = y_test_pred_proba[:, 0]  # Probability of Model A winning
    submission_df['winner_model_b'] = y_test_pred_proba[:, 1]  # Probability of Model B winning
    submission_df['winner_model_tie'] = y_test_pred_proba[:, 2]  # Probability of a tie

    print(submission_df.head())
    return submission_df


def create_submission_file(submission_df, filename='sample_submission.csv'):
    submission_df.to_csv(filename, index=False)
    print(f'Submission file saved as {filename}')


def main():
    # Load dataset
    train_df, test_df = load_data('data/lmsys-chatbot-arena.zip')

    # Add basic features like response length
    train_df = add_basic_features(train_df)
    test_df = add_basic_features(test_df)

    # Prepare data for training
    X_train, X_val, y_train, y_val = prepare_data(train_df)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_val, y_val)

    # Make predictions on the test set
    submission = make_predictions(model, test_df)

    # Create and save the submission file
    create_submission_file(submission)


if __name__ == '__main__':
    main()