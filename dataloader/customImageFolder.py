from torchvision.datasets import ImageFolder


class PathImageFolder(ImageFolder):

    def __getitem__(self, index):
        # Getting the original tuple
        original_tuple = super(PathImageFolder, self).__getitem__(index)

        # Gettin the path of current image
        path = self.imgs[index][0]

        # Creating a tuple containing path
        path_tuple = original_tuple + (path,)

        return path_tuple
