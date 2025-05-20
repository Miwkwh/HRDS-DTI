import torch
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, confusion_matrix, roc_curve, \
    recall_score, accuracy_score, precision_recall_curve, matthews_corrcoef
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator


class Trainer(object):
    def __init__(self, model, optim, celoss, device, train_dataloader, val_dataloader, test_dataloader,
                 configs, output_path, scheduler, logger):
        self.model = model
        self.optim = optim
        self.CEloss = celoss
        self.device = device
        self.schedular = scheduler
        self.logger = logger

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.n_class = configs.MLP.Binary
        self.epochs = configs.Global.Epoch
        self.current_epoch = 0
        self.batch_size = configs.Global.Batch_Size

        self.float2str = lambda x: '%0.4f' % x if isinstance(x, (float, np.float32, np.float64)) else str(x)
        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.writer = None
        self.configs = configs
        self.output_dir = output_path

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "Accuracy", "Recall", "Precision", "Cross Matrix",
                              "MCC", "F1", "Sensitivity", "Specificity", "Test_loss"]
        train_metric_header = ["# Epoch", "Train_loss"]

        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

    def set_tensorboard(self, path):
        model_name = f'{path}/tensorboard_output'
        self.writer = SummaryWriter(log_dir=model_name)

    def train(self):
        for i in range(1, self.epochs + 1):
            self.current_epoch += 1
            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(self.float2str, [train_loss]))

            self.train_loss_epoch.append(train_loss)
            self.train_table.add_row(train_lst)

            val_auroc, val_auprc, val_loss = self.test(dataloader="val")
            self.schedular.step(val_auroc) 


            val_lst = ["epoch " + str(self.current_epoch)] + list(map(self.float2str, [val_auroc, val_auprc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(val_auroc)
            if val_auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = val_auroc
                self.best_epoch = self.current_epoch
            self.logger.info(f'Validation loss:{val_loss:.4f} AUROC:{val_auroc:.4f} AUPRC:{val_auprc:.4f}')
        
            if self.writer:
                scalars_loss = {'loss_train': train_loss, 'loss_valid': val_loss}
                self.writer.add_scalars('epoch/loss', scalars_loss, self.current_epoch)
                self.writer.add_scalar('epoch/auroc_valid', val_auroc, self.current_epoch)
                self.writer.add_scalar('epoch/auprc_valid', val_auprc, self.current_epoch)

        if self.writer:
            self.writer.close()

     
        auroc, auprc, accu, recall, precision, cm, mcc, f1, sensitivity, specificity, test_loss = self.test(
            dataloader="test") 



        test_lst = ["epoch " + str(self.best_epoch)] + list(map(self.float2str, [auroc, auprc, accu, recall,
                                                                                 precision, cm, mcc, f1, sensitivity,
                                                                                 specificity, test_loss]))
        self.test_table.add_row(test_lst)

        self.logger.info(
            f"Test Results Best Epoch {self.best_epoch}:\n"
            f"  Test Loss: {test_loss:.4f}\n"
            f"  AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}\n"
            f"  Accuracy: {accu:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}"
        )
        self.logger.info(f"Confusion Matrix:\n{cm}")
        self.logger.info(
            f"Metrics Summary:\n"
            f"  MCC: {mcc:.4f}, F1: {f1:.4f}\n"
            f"  Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}"
        )


        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["accuracy"] = accu
        self.test_metrics['mcc'] = mcc
        self.test_metrics['f1'] = f1
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["precision"] = precision
        self.test_metrics["recall"] = recall

        self.save_result()
        return self.test_metrics

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for batch_index, (v_d, v_p, labels) in enumerate(self.train_dataloader):
            v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.to(self.device)

            self.optim.zero_grad()
            v_d, v_p, f, score = self.model(v_d, v_p)
            loss = self.CEloss(score, labels.long())

            loss.backward()
            self.optim.step()


            loss_epoch += loss.item()

        loss_epoch = loss_epoch / num_batches
        self.logger.info(f'Training at Epoch {self.current_epoch} with training loss {loss_epoch:.4f}')
        return loss_epoch

    def test(self, dataloader="test"):
        test_loss = 0
        Y, P, S = [], [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)

        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, labels) in enumerate(data_loader):
                v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.to(self.device)
                if dataloader == "val":
                    v_d, v_p, f, score = self.model(v_d, v_p)
                elif dataloader == "test":
                    v_d, v_p, f, score = self.best_model(v_d, v_p)
                loss = self.CEloss(score, labels.long())
                y_labels = labels.to('cpu').data.numpy()
                y_scores = F.softmax(score, 1).to('cpu').data.numpy()
                y_preds = np.argmax(y_scores, axis=1)
                y_scores = y_scores[:, 1]

                test_loss += loss.item()
                Y.extend(y_labels)
                P.extend(y_preds)
                S.extend(y_scores)

        auroc = roc_auc_score(Y, S)
        auprc = average_precision_score(Y, S)
        correct_preds = torch.sum(torch.tensor(Y) == torch.tensor(P))
        test_loss = test_loss / num_batches
        self.logger.info(f'Val: correct prediction is {correct_preds}/{len(Y)} = '
              f'{correct_preds.double()/len(Y)*100:.2f}%')

        if dataloader == "test":
            accuracy = accuracy_score(Y, P)
            recall = recall_score(Y, P)
            precision = precision_score(Y, P)
            f1 = 2 * (precision * recall) / (precision + recall)
            mcc = matthews_corrcoef(Y, P)
            cm = confusion_matrix(Y, P)
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
            return auroc, auprc, accuracy, recall, precision, cm, mcc, f1, sensitivity, specificity, test_loss
        else:
            return auroc, auprc, test_loss

    def save_result(self):
        if self.configs.Result.Save_Model:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, "best_model_epoch.pt"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.configs
        }
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))
        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())
