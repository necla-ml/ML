import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
from ml import io

def save(fig, path, format='png'):
    figure = plt.gcf()
    plt.figure(fig.number)
    plt.savefig(path, format=format)
    plt.figure(figure.number)

def show(fig):
    figure = plt.gcf()
    plt.figure(fig.number)
    plt.show()
    plt.figure(figure.number)

def numpy(fig, dpi=None):
    from ml import cv, math
    bio = io.BytesIO()
    fig.savefig(bio, format='raw', dpi=dpi or fig.dpi)
    bio.seek(0)
    img = np.frombuffer(bio.getvalue(), dtype=np.uint8)
    size = fig.get_size_inches() * (dpi or fig.dpi)
    img = np.reshape(img, (*size.astype('int'), -1))
    bio.close()
    return img

def confusion_matrix(cm, labels=None, title=None, size=8, dpi=96):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
        cm(Tensor[K, K]): a confusion matrix of integer classes
    Kwargs:   
        labels(List[K]): String names of the integer classes
        title(str):
        size(Number or tuple(Number, Number)): in inches
    """
    if labels is None:
        labels = [str(c) for c in range(cm.shape[0])]
    
    import matplotlib.pyplot as plt
    if isinstance(size, int):
        size = (size, size)
    figure = plt.figure(figsize=size, dpi=dpi)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title or "Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    # Normalize the confusion matrix.
    import torch
    if torch.is_tensor(cm):
        cm = cm.numpy()
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.ylabel('Ground Truth')
    plt.xlabel('Predictions')
    plt.tight_layout()
    return figure