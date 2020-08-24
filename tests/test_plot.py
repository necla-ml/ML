import pytest
from ml import plot, logging

@pytest.mark.essential
def test_confusion_matrix():
    import torch
    from sklearn.metrics import confusion_matrix
    truth = torch.arange(10)
    perm = torch.randperm(len(truth))
    preds = truth[perm]
    cm = confusion_matrix(truth, preds)
    fig = plot.confusion_matrix(cm)
    img = plot.numpy(fig)
    size = fig.get_size_inches() * fig.dpi
    assert img.shape == (*size.astype('int'), 4)
    if False:
        from ml import cv
        img = cv.cvtColor(img, cv.COLOR_RGBA2BGR)
        cv.save(img, 'cm.jpg')
        plot.save(fig, 'cm.png', format='png')