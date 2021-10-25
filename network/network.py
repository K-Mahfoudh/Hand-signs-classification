import torch
from torchvision import models
from torch import nn, optim
import time
import torch.nn.functional as F
import numpy as np
from visualization import visualize_image
import sys
from itertools import islice


class Network(nn.Module):
    """
    Implementation of the neural network.

    """
    def __init__(self, model_path, epochs, lr, slice):
        """
        Neural network constructor

        :param model_path: string representing path to the model (for saving or loading and existent model)
        :param epochs:  integer representing the number of epochs to be used for training
        :param lr: float value representing the learning rate to be used in training
        :param slice: an integer representing the slice of the training set to be used in training, this param is used
        only because i didn't have a good GPU, so i performed training on a part of the data, saved the model, then
        performed the training on the whole data based on the previous model
        """
        super(Network, self).__init__()
        self.criterion = nn.NLLLoss()
        self.optimizer = None
        self.model_path = model_path
        self.model = models.resnet152(pretrained=True, progress=True)
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_valid_loss = np.inf
        self.set_classifier(lr)
        self.slice = slice

    def set_classifier(self, lr):
        """
        A method used to change classification layer (fine-tuning).

        :param lr: learning rate to be used in training
        """
        # Disable parameters training for the model
        for parameters in self.model.parameters():
            parameters.requires_grad = False

        # Changing model classifier
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.45),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.45),
            nn.Linear(128, 28),
            nn.LogSoftmax(dim=1)
        )

        # Changing optimizer
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=lr)


    def forward(self, data):
        """
        Implementation of forward propagation.

        :param data: DataLoader instance representing data to be used in training
        :return: result of forwarding data in the neural network (logps)
        """
        # Performing first convolution (images are in grayscale, so depth is 1, but needed depth is 3)
        #data = self.conv0(data)
        return self.model(data)

    def get_model_details(self):
        """
        Helper method used to print information about the neural network.

        """
        print(self.model)

    def train_network(self, train_loader, valid_loader):
        """
        Method used to train the neural network and calculate the loss and accuracy of training and validation.

        This method contains 2 parts, the first part is training, where we perform forward and back propagation and
        we update the parameters of the model. Second part is validation, we store parameters that gives the
        best (minimum) loss. Note that the program will train the model using GPU if it supports cuda, otherwise,
        it will use CPU.

        :param train_loader: dataloader object containing training images
        :param valid_loader: dataloader object containing validation images
        :return: 4 lists containing history of loss and accuracy of both training and validation steps
        """
        train_loss_list = []
        train_accuracy_list = []
        valid_loss_list = []
        valid_accuracy_list = []

        # Sending model to GPU if available
        self.to(self.device)

        for epoch in range(self.epochs):
            print('Epoch: {}'.format(epoch))
            # Starting time of epoch
            epoch_start_time = time.time()

            # Init accuracy and loss
            train_accuracy, valid_accuracy, train_loss, valid_loss = 0, 0, 0, 0

            # Enable train mode
            self.train()
            for index, (images, labels, paths) in enumerate(islice(train_loader,0,len(train_loader)//self.slice)):
                # Sending images to GPU if cuda is available
                images, labels = images.to(self.device), labels.to(self.device)

                # Reset grads
                self.optimizer.zero_grad()

                # Performing forward pass
                logits = self.forward(images)

                # Calculating loss
                loss = self.criterion(logits, labels)
                train_loss += loss

                # backward propagation
                loss.backward()

                # Updating gradients
                self.optimizer.step()

                # Getting predictions
                preds = F.softmax(logits, dim=1)

                # Getting predicted class
                _, top_class = preds.topk(1, dim=1)

                # Comparing between classes and labels
                compare = top_class == labels.view(*top_class.shape)

                # Calculating accuracy
                train_accuracy += torch.mean(compare.type(torch.FloatTensor))

                # Printing train loss
                sys.stdout.write('Batch :{}/{} ---- Train loss: {:.3f}\r'.format(index, len(train_loader)//self.slice, loss))
                sys.stdout.flush()

            # Switching to evaluation mode
            self.eval()
            with torch.no_grad():
                for index, (images, labels, paths) in enumerate(islice(valid_loader, 0, len(valid_loader)//self.slice)):
                    # Sending images and labels to GPU if cuda is available
                    images, labels = images.to(self.device), labels.to(self.device)

                    # Forward propagation
                    logits = self.forward(images)

                    # Calculating loss
                    loss = self.criterion(logits, labels)
                    valid_loss += loss

                    # Getting predictions
                    preds = F.softmax(logits, dim=1)


                    # Getting predicted classes
                    top_p, top_class = preds.topk(1, dim=1)


                    # Comparing predictions and labels
                    compare = top_class == labels.view(*top_class.shape)

                    valid_accuracy += torch.mean(compare.type(torch.FloatTensor))
                    sys.stdout.write('Batch :{}/{} ---- Validation loss: {:.3f}\r'.format(index, len(valid_loader)//self.slice, loss))
                    sys.stdout.flush()

            # Claculating loss and accuracy for train and validation sets
            train_accuracy = train_accuracy/len(train_loader)*100*self.slice
            train_loss = train_loss/len(train_loader)*self.slice
            valid_accuracy = valid_accuracy/len(valid_loader)*100*self.slice
            valid_loss = valid_loss/len(valid_loader)*self.slice

            # Appending values to list (for visualization purpose)
            train_accuracy_list.append(train_accuracy)
            train_loss_list.append(train_loss)
            valid_accuracy_list.append(valid_accuracy)
            valid_loss_list.append(valid_loss)

            # Creating a checkpoit if valid loss is reduced
            if valid_loss < self.min_valid_loss:
                print('Validation loss decreased from {:.3f} =======> {:.3f}\r'.format(self.min_valid_loss, valid_loss))
                self.min_valid_loss = valid_loss
                print('Saving model in path: {}'.format(self.model_path))
                torch.save({
                    'state_dict': self.state_dict(),
                    'min_loss': self.min_valid_loss,
                    'classifier': (2048, 512, 128, 29),
                    'input_size': 224,
                    'optimizer': self.optimizer.state_dict()
                }, self.model_path)
            print(('Epoch: {}-{:.3f} =====>  Train Accuracy: {:.3f} ------' +
                  'Train Loss: {:.3f} ------ Valid Accuracy: {:.3f} ------ Valid Loss: {:.3f} \r').format(
                    epoch,
                    time.time()-epoch_start_time,
                    train_accuracy,
                    train_loss,
                    valid_accuracy,
                    valid_loss))

        return train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list

    def predict(self, dataset):
        """
        A method used to predict the labels of given images, and print accuracy and loss.

        :param dataset: dataloader object containing test images.
        """
        self.to(self.device)
        if not self.eval():
            self.eval()
        with torch.no_grad():
            loss = 0
            accuracy = 0
            for index, (images, labels, paths) in enumerate(dataset):
                # Sending data to GPU if cuda is available
                if len(images) > 1:
                    images, labels = images.to(self.device), labels.to(self.device)

                # Performing forward pass
                logits = self.forward(images)

                # Calculating loss
                loss += self.criterion(logits, labels)

                # Getting predictions
                preds = F.softmax(logits, dim=1)

                # Getting top class and top probabilities
                top_p, top_class = preds.topk(1, dim=1)

                # comparing between labels
                compare = top_class == labels.view(*top_class.shape)

                # Calculating accuracy
                acc = torch.mean(compare.type(torch.FloatTensor))
                print('Batch accuracy is: {}'.format(acc))
                accuracy += acc

                # Visualization
                visualize_image(images, labels, paths, top_class, top_p)

            accuracy = accuracy / len(dataset) * 100
            loss = loss / len(dataset)
            print('The accuracy is {} ------- loss: {}'.format(accuracy, loss))

    def load_model(self, model_path):
        """
        Method used to load an existent model.

        :param model_path: path to the model to be loaded
        """
        print('Loading model from path: {}'.format(model_path))
        model_dict = torch.load(model_path)
        self.min_valid_loss = model_dict['min_loss']
        self.load_state_dict(model_dict['state_dict'])
        print('Min valid loss is: {}'.format(self.min_valid_loss))

    """
    def NLLLoss(self, logits, labels):
            output = torch.zeros_like(labels)
            for i in range(len(labels)):
                output[i] = logits[i][labels[i]]

            return -output.type(torch.FloatTensor).sum()/len(output)
    

    def set_min_valid_loss(self, loss):
        self.min_valid_loss = loss
    """