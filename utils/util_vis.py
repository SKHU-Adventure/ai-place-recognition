import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def draw_roc_curve(fpr, tpr, thresholds, best_threshold, save_path='./roc_curve.png', roc_auc=None):
    plt.figure()
    lw = 2
    closest_index = np.argmin(np.abs(thresholds - best_threshold))
    
    plt.plot(fpr[closest_index], tpr[closest_index], 'ro', label=f'TPR at threshold {-best_threshold:.2f}: {tpr[closest_index]:.2f}')
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)

def draw_confusion_matrix(cm, threshold, save_path='./confusion_matrix.png'):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix at Threshold {-threshold:.4f}')
    plt.savefig(save_path)

def find_best_threshold(fpr, tpr, thresholds):
    j_scores = tpr - fpr
    best_index = np.argmax(j_scores)
    best_threshold = thresholds[best_index]
    return best_threshold