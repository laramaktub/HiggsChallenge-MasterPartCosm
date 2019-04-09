""" Module with tools to plot and study the training history of the neural network """


import keras.callbacks as cb
import matplotlib.pyplot as plt

class LossHistory(cb.Callback):

    """
    Class to store the training information: loss and accuracy
    """

    def on_train_begin(self, logs={}):
	
        # Initialize the lists for holding the logs, losses and accuracies:
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
    
    def on_epoch_end(self, batch, logs={}):
        
        # This function is called at the end of each epoch

        # Append the logs, losses and accuracies to the lists:
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        
        
        
def plot_history(history, Num_classes, n_batch, n_epochs):
   
    """
    Plot the history atributes of the training procedure:
    
    - Training loss
    - Training accuracy
    - Validation loss
    - Validation accuracy

    """

    # Get the history values
    loss_list = history.losses
    acc_list = history.acc
    val_loss_list = history.val_losses
    val_acc_list = history.val_acc

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    # Get the epochs:
    epochs = range(1,len(loss_list) + 1)
   
    # Define the frame figure characteristics:
    plt.figure(num=None, figsize=(6, 10))
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)
    
    # First subplot: LOSS 
    plt.subplot(211)
    plt.plot(epochs, loss_list, 'b', label='Training loss')
    plt.plot(epochs, val_loss_list, 'g', label = 'Validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Second subplot: ACCURACY
    plt.subplot(212)
    plt.plot(epochs, acc_list, "b", label = 'Training accuracy')
    plt.plot(epochs, val_acc_list, 'g', label = 'Validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
   
    # Save the History:
    plt.savefig("History/History_"+str(Num_classes)+"classes_"+str(n_batch)+"batch_"+str(n_epochs)+"_epochs.png", dpi = 600)
    plt.savefig("History/History.png", dpi = 600)