from segment_utils.labels import right_tool_labels, left_tool_labels
from sklearn.metrics import f1_score, accuracy_score


def f1_acc(actual_right, actual_left, pred_right, pred_left):
    right_f1 = f1_score(
        y_true=actual_right,
        y_pred=pred_right,
        labels=right_tool_labels,
        average='macro'
    )
    left_f1 = f1_score(
        y_true=actual_left,
        y_pred=pred_left,
        labels=left_tool_labels,
        average='macro'
    )
    f1 = (right_f1 + left_f1) / 2
    acc = accuracy_score(
        y_true=actual_right + actual_left,
        y_pred=pred_right + pred_left,
    )

    return f1, acc
