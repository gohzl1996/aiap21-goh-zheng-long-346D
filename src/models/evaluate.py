from typing import Dict
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(y_true, y_pred) -> Dict:
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return {"confusion_matrix": cm, "report": report}
