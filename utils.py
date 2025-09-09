from sklearn import metrics

def get_metrics(labels_true, labels_pred, labels_to_ids):
    cls_report = metrics.classification_report(labels_true, labels_pred, target_names=labels_to_ids,
                                               labels=tuple(labels_to_ids.values()), zero_division=0.0, digits=5)
    cls_report_dict = metrics.classification_report(labels_true, labels_pred, target_names=labels_to_ids,
                                                    labels=tuple(labels_to_ids.values()), zero_division=0.0, output_dict=True)
    
    macro_f1 = metrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')[source]

    accuracy, macro_f1 = cls_report_dict['accuracy'], cls_report_dict['macro avg']['f1-score']
    return cls_report, accuracy, macro_f1