import matplotlib.pyplot as plt
import numpy as np

def draw_roc_curve(fpr, tpr, thresholds, save_path='./roc_curve.png', roc_auc=None):
    plt.figure()
    lw = 2
    closest_index = np.argmin(np.abs(thresholds + 1))
    
    plt.plot(fpr[closest_index], tpr[closest_index], 'ro', label=f'TPR at threshold 1.0: {tpr[closest_index]:.2f}')
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    