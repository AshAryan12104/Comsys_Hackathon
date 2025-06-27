from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_gender_metrics(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }

def compute_identity_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro')
    }

