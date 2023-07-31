import io
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

plt.rcParams.update({'figure.max_open_warning': 2})

def set(param, value):
    matplotlib.rcParams[param] = value

def fignums():
    return plt.get_fignums()

def close(fig=None):
    plt.close(fig)

def save(fig, path, format='png', close=True):
    gcf = plt.gcf()
    bak = fig is not gcf
    if bak:
        plt.figure(fig.number)
    from pathlib import Path
    path = Path(path)
    format = format == path.suffix and format or path.suffix[1:]
    plt.savefig(path, format=format)
    if close:
        plt.close(fig)
    if bak:
        plt.figure(gcf.number)

def show(fig):
    gcf = plt.gcf()
    bak = fig is not gcf
    if bak:
        plt.figure(fig.number)
    plt.show()
    if bak:
        plt.figure(figure.number)

def numpy(fig, dpi=None):
    bio = io.BytesIO()
    fig.savefig(bio, format='raw', dpi=dpi or fig.dpi)
    bio.seek(0)
    img = np.frombuffer(bio.getvalue(), dtype=np.uint8)
    size = fig.get_size_inches() * (dpi or fig.dpi)
    img = np.reshape(img, (*size.astype('int'), -1))
    bio.close()
    return img

def confusion_matrix(cm, labels=None, title=None, size=8, dpi=100, interpolation='none', cmap='viridis'):
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
    if isinstance(size, int):
        size = (size, size)

    #plt.cla()
    #plt.clf()
    #plt.close()
    figure = plt.figure(figsize=size, dpi=dpi)
    
    # Round the confusion matrix.
    import torch
    if torch.is_tensor(cm):
        cm = cm.numpy()
    cm = np.around(cm.astype('float'), decimals=2)
    
#    plt.imshow(cm, interpolation=interpolation, cmap=plt.cm.Blues)
    plt.imshow(cm, interpolation=interpolation, cmap=plt.get_cmap(cmap))
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation='vertical', fontsize=8)
    plt.yticks(tick_marks, labels, fontsize=8)
    plt.title(title or "Confusion Matrix")
    plt.grid()

    # Draw white text if squares are dark; otherwise black only for small matrix
    if len(cm) < 51:
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.ylabel('Ground Truth')
    plt.xlabel('Predictions')
    plt.tight_layout()
    return figure