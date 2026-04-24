from pure_cifar_10 import CIFAR10

loader = CIFAR10()   # create loader

train_images, train_labels, test_images, test_labels = loader.load_all()

