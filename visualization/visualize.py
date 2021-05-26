import matplotlib.pyplot as plt
from data import get_class_name


def visualize_num(train_loss, train_accuracy, valid_loss, valid_accuracy):
    # Visualizing loss and accuracy of train and valid sets

    fig = plt.figure(figsize=(64, 32))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(train_loss, label='Train loss')
    ax.plot(train_accuracy, label='Train accuracy')
    ax.plot(valid_loss, label='Validation loss')
    ax.plot(valid_accuracy, label='Validation_accuracy')
    plt.show()


def visualize_image(images, labels, paths, preds, top_p):
    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()
    top_p = top_p.cpu()
    print('Visualizing data')
    # Creating a figure
    fig = plt.figure(figsize=(32, 4))
    for index in range(len(images)):
        ax = fig.add_subplot(9, 32/8, index+1)
        ax.imshow(images[index].numpy().transpose())
        ax.set_title(str(labels[index].item()) + '---' + get_class_name(paths[index], -2)
                     + '---Preds: {}'.format(*preds[index].numpy())
                     + '---Top_p: {:.3f}'.format(*top_p[index].numpy()))
        fig.subplots_adjust(wspace=5)
    plt.show()