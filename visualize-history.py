import pickle

########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################


history = pickle.load( open( "/Users/emanueledifiore/Dropbox/Università/Materiale Didattico/Magistrale/Tesi/Autoencoders/Anomaly Detection with Autoencoders/DCASE2020Challenge/conditioned_autoencoder_190221/trainHistoryDict", "rb" ) )

#print(history['val_mse'])
visualizer = visualizer()
visualizer.loss_plot(loss = history["loss"], val_loss = history['val_mse'])
visualizer.save_figure("./history.png")