import matplotlib.pyplot as plt

def draw_roc_curve(fpr, tpr, save_path='./roc_curve.png', roc_auc=None):
    plt.figure()
    lw = 2
    label = 'ROC curve (area = %0.2f)' % roc_auc if roc_auc is not None else 'ROC curve'
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=label)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    