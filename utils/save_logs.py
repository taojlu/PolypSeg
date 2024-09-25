

# save model params
class SaveModelParam(object):
    def __init__(self, model_path, timestamp, model_name, model_params, loss_name, batch_size,
                 optimizer, learning_rate, augmentation):
        super(SaveModelParam, self).__init__()
        self.model_path = model_path
        self.timestamp = timestamp
        self.model_name = model_name
        self.model_params = model_params
        self.loss_name = loss_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.augmentation = augmentation

    def write_model(self):
        with open('{}/{}.log'.format(self.model_path, self.timestamp), 'a', newline='') as f:
            f.write("Model Name: {}".format(self.model_name))
            f.write("\n")
            f.write("Model Params: {}".format(self.model_params))
            f.write("\n")
            f.write("Loss Function: {}".format(self.loss_name))
            f.write("\n")
            f.write("Batch Size: {}".format(str(self.batch_size)))
            f.write("\n")
            f.write("Augmentation: {}".format(str(self.augmentation)))
            f.write("\n")
            f.write("Learning Rate: {}".format(str(self.learning_rate)))
            f.write("\n")
            f.write("Optimizer: {}".format(str(self.optimizer)))
            f.write("\n")
            print('-----------------------------------------\n', file=f)


# save model params
class SaveTrainingMetrics(object):
    def __init__(self, model_save_path, timestamp, epoch, train_loss, CVC_300_dice, CVC_ClinicDB_dice, Kvasir_dice,
                 CVC_ColonDB_dice, ETIS_LaribPolypDB_dice):
        super(SaveTrainingMetrics, self).__init__()
        self.model_path = model_save_path
        self.timestamp = timestamp
        self.epoch = epoch
        self.train_loss = train_loss
        self.CVC_300_dice = CVC_300_dice
        self.CVC_ClinicDB_dice = CVC_ClinicDB_dice
        self.Kvasir_dice = Kvasir_dice
        self.CVC_ColonDB_dice = CVC_ColonDB_dice
        self.ETIS_LaribPolypDB_dice = ETIS_LaribPolypDB_dice

    def write_metrics(self):
        with open('{}/{}.log'.format(self.model_path, self.timestamp), 'a', newline='') as f:
            f.write("-----------------------------------------")
            f.write("\n")
            f.write("Epoch: {}".format(self.epoch))
            f.write("\n")
            f.write("Train Loss: {}".format(self.train_loss))
            f.write("\n")
            f.write("Kvasir dice: {:.4f}".format(self.Kvasir_dice))
            f.write("\n")
            f.write("ETIS-Larib dice: {:.4f}".format(self.ETIS_LaribPolypDB_dice))
            f.write("\n")
            f.write("CVC-ColonDB dice: {:.4f}".format(self.CVC_ColonDB_dice))
            f.write("\n")
            f.write("CVC-ClinicDB dice: {:.4f}".format(self.CVC_ClinicDB_dice))
            f.write("\n")
            f.write("CVC-300 dice: {:.4f}".format(self.CVC_300_dice))
            f.write("\n")


