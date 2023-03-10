
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, writer=None):
        self.name = name
        self.writer = writer
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def summary(self, epoch=None, some_txt=''):
        if self.writer is not None:
            #print(f"writing {self.name}")
            self.writer.add_scalar(self.name, self.avg, epoch)
        return f'{self.name} at epoch {epoch} [{some_txt}] => {self.avg}'


class BinaryAccuracy(object):
    """Computes and stores the accuracy for binaru classication"""
    def __init__(self, name, writer=None):
        self.name = name
        self.writer = writer
        self.reset()

    def reset(self):
        self.all_pred = []
        self.all_target = []

    def update(self, pred_logits, target, threshold = 0):

        # convert to the predictions
        pred = np.zeros(pred_logits.shape[0])
        # print(pred.shape, pred_logits.shape)
        pred[pred_logits > threshold] = 1

        # collect all labels
        self.all_pred.append(pred)
        self.all_target.append(target)
        

    def summary(self, epoch=None, some_txt=''):

        all_pred = np.concatenate(self.all_pred)
        all_target = np.concatenate(self.all_target)

        cm = confusion_matrix(all_target, all_pred, labels=[0,1], normalize='true')
        tn, fp, fn, tp = cm.ravel()

        ba = (tp + tn)*0.5
        if self.writer is not None:
            self.writer.add_scalar(f'{self.name}/TNR', tn, epoch)
            self.writer.add_scalar(f'{self.name}/TPR', tp, epoch)
            self.writer.add_scalar(f'{self.name}/BAcc', ba, epoch)
            

        #return f'{tn_str} {fp_str} {fn_str} {tp_str}'
        return f'{self.name} at epoch {epoch} [{some_txt}] =>: Bacc: {ba}, TNR: {tn},  TPR: {tp}'


class BinaryAccuracyWithCat(object):
    """Computes and stores the accuracy for binaru classication"""
    def __init__(self, name, writer=None, categories=['box', 'bottle', 'bowl', 'mug', 'cylinder', 'spatula', 'hammer', 'pan', 'fork', 'scissor']):
        self.name = name
        self.writer = writer
        self.categories = categories
        self.reset()

    def reset(self):
        self.all_pred = {}
        self.all_target = {}
        for cat in self.categories:
            self.all_pred[cat] = []
            self.all_target[cat] = []

    def update(self, pred_logits, target, cat, threshold = 0):

        # convert to the predictions
        pred = np.zeros(pred_logits.shape[0])
        # print(pred.shape, pred_logits.shape)
        pred[pred_logits > threshold] = 1


        # collect all labels
        for i in range(len(cat)):
            self.all_pred[cat[i]].append(pred[i])
            if target[i] >= 0.5:
                self.all_target[cat[i]].append(1)
            else:
                self.all_target[cat[i]].append(0)
        
    def summary(self, epoch=None, some_txt=''):

        
        print(f'{self.name} at epoch {epoch} [{some_txt}]:')
        all_pred = []
        all_target = []
        for cat in self.all_pred.keys():
            all_pred.append(self.all_pred[cat])
            all_target.append(self.all_target[cat])

            cm = confusion_matrix(self.all_target[cat], self.all_pred[cat], labels=[0,1], normalize='true')
            tn, fp, fn, tp = cm.ravel()
            ba = (tp + tn)*0.5
            print(f'{cat} =>: Bacc: {ba}, TNR: {tn},  TPR: {tp}')

            
        all_pred = np.concatenate(all_pred)
        all_target = np.concatenate(all_target)

        cm = confusion_matrix(all_target, all_pred, labels=[0,1], normalize='true')
        tn, fp, fn, tp = cm.ravel()
        
        ba = (tp + tn)*0.5
        if self.writer is not None:
            self.writer.add_scalar(f'{self.name}/TNR', tn, epoch)
            self.writer.add_scalar(f'{self.name}/TPR', tp, epoch)
            self.writer.add_scalar(f'{self.name}/BAcc', ba, epoch)

        print(f'Total =>: Bacc: {ba}, TNR: {tn},  TPR: {tp}')