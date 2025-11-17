def macro_f1(report_dict):
    return report_dict["macro avg"]["f1-score"]

def per_class_f1(report_dict):
    return {k: v["f1-score"] for k, v in report_dict.items() if k not in {"accuracy", "macro avg", "weighted avg"}}
