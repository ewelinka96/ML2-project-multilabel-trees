import sklearn.metrics as slm
import pandas as pd

def metrics(y_true, y_pred, model_type = "Model"):
    confusion = slm.confusion_matrix(y_true, y_pred)
    TP = confusion[1, 1] # True Positives
    TN = confusion[0, 0] # True Negatives
    FP = confusion[0, 1] # False Positives
    FN = confusion[1, 0] # False Negatives
    result_df = pd.DataFrame({model_type:
                                      [slm.accuracy_score(y_true, y_pred),
                                      slm.balanced_accuracy_score(y_true, y_pred),
                                      TP / float(TP + FP),
                                      TP / float(TP + FN),
                                      TN / float(TN + FP),
                                      (2*TP)/float(2*TP + FP + FN),
                                     ]},
                             index=["Accuracy",
                                    "Balanced accuracy",
                                    "Precision",
                                    "Sensitivity",
                                    "Specificity",
                                    "F1 score"])
    return confusion, result_df
    #print("Label ranking average precision: ", slm.label_ranking_average_precision_score(y_true, y_pred))
    #print("Ranking loss: ", slm.label_ranking_loss(y_true, y_pred))
