from data import Data
from network import Network
from visualization import visualize_image, visualize_num
import argparse
import os
from hand_tracking import HandTracker


def main(args):
    # Creating data object
    data = Data(args.train_path, args.test_path, args.batch_size, args.valid_size)

    if args.mode == 'Train':
        # Loading train and validation sets
        train_loader, valid_loader = data.get_trainset()

        # Creating neural network
        network = Network(model_path=args.model_save_path, epochs=args.epochs, lr=args.learning_rate, slice=args.slice)

        # If checkpoint is set to True, the selected model is loader and training will proceed
        if args.checkpoint and os.path.exists(args.model_load_path):
            network.load_model(args.model_load_path)

        # Training network
        train_loss, train_accuracy, valid_loss, valid_accuracy = network.train_network(train_loader, valid_loader)

        # Visualizing Accuracy and loss for train and validation sets
        visualize_num(train_loss, train_accuracy, valid_loss, valid_accuracy)
    else:
        # Creating network
        network = Network(model_path=args.model_save_path, epochs=args.epochs, lr=args.learning_rate, slice=args.slice)
        # Loading model
        if os.path.exists(args.model_load_path):
            network.load_model(args.model_load_path)

        if args.mode == 'Test':
            # Getting test set
            test_data = data.get_testset(shuffle=args.shuffle)

            # Predicting
            network.predict(test_data, False)

        # Main program
        elif args.mode == 'Main':
            hand_tracker = HandTracker(network)
            hand_tracker.capture_video(True)

        else:
            raise Exception('Wrong mode selected, try Train or Test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' Training a neural network for Hand sign recognition')

    parser.add_argument('-m', '--mode', help='Train for training network, Test for testing network', default='Test')
    parser.add_argument('-ms', '--model_save_path', help='Path for saving the trained model',
                        default='models/resnet-152-v7-224.pth')
    parser.add_argument('-c',
                        '--checkpoint',
                        help='If set to True, training starts from previously trained model',
                        type=bool,
                        required=False,
                        default=False)
    parser.add_argument('-l',
                        '--model_load_path',
                        help='Loading previous checkpoint to continue training, must be set if checkpoint is True',
                        required=False, default='models/resnet-152-v4-224.pth')
    parser.add_argument('-tp', '--train_path', help='Path to training data', default='data/signs_data/train')
    parser.add_argument('-ts', '--test_path', help='Path to testing data', default='data/signs_data/test')
    parser.add_argument('-b', '--batch_size', type=int, help='The batch size to be used in training', default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, help='Selecting learning rate', default=0.0003)
    parser.add_argument('-e', '--epochs', help='Number of training epochs', type=int, default=250)
    parser.add_argument('-v', '--valid_size', help='float representing size of validation set portion',
                        type=float,
                        default=0.2)
    parser.add_argument('-s', '--slice',
                        help='Defines size of data used in training, this is used of you have a bad GPU and huge data',
                        type=int,
                        default=1)
    parser.add_argument('-sh', '--shuffle', type=bool, help='Set to True for shuffling test data', required=False,
                        default=True)
    args = parser.parse_args()

    main(args)
