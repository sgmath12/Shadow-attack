import torchvision
import torchvision.transforms as transforms
import pdb

def MNIST(root = 'D:\\dataset',normalize = False,download = False):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = torchvision.datasets.MNIST(root = root, train=True, transform=transform, target_transform=None, download=download)
    val_set = torchvision.datasets.MNIST(root = root, train=False, transform=transform, target_transform=None, download=download)

    return train_set,val_set


def FashionMNIST(root = 'D:\\dataset',normalize = False):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = torchvision.datasets.MNIST(root = root, train=True, transform=transform, target_transform=None, download=True)
    val_set = torchvision.datasets.MNIST(root = root, train=False, transform=transform, target_transform=None, download=True)

    return train_set,val_set

def CIFAR10(root = 'D:\\dataset',normalize = False,download = False):
    if normalize == True:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    train_set = torchvision.datasets.CIFAR10(root = root, train=True, transform = transform, target_transform=None, download=download)
    val_set = torchvision.datasets.CIFAR10(root = root , train=False, transform = transform, target_transform=None, download=download)
    
    return train_set,val_set

def CIFAR100(root = 'D:\\dataset',normalize = False):
    if normalize == True:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    train_set = torchvision.datasets.CIFAR100(root = root, train=True, transform = transform, target_transform=None, download=False)
    val_set = torchvision.datasets.CIFAR100(root = root , train=False, transform = transform, target_transform=None, download=True)
    
    return train_set,val_set

def IMAGENET(root = 'D:\\dataset',normalize = False):

    if normalize == False:
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        
    else:
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    train_set = torchvision.datasets.ImageFolder(root + 'train', transform = transform)
    val_set = torchvision.datasets.ImageFolder(root + 'val', transform = transform)

    return train_set,val_set