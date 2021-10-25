from torchvision.datasets import ImageFolder


class PathImageFolder(ImageFolder):
    """
    A custom Image folder that returns in addition to images, the path of each image. This class is used for
    visualization purposes.

    """
    def __getitem__(self, index):
        """
        Redefinition of getitem method.

        :return: a custom ImageFolder containing images paths
        """
        # Getting the original tuple
        original_tuple = super(PathImageFolder, self).__getitem__(index)

        # Gettin the path of current image
        path = self.imgs[index][0]

        # Creating a tuple containing path
        path_tuple = original_tuple + (path,)

        return path_tuple
