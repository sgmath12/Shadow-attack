  
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import dataset
import torchvision
import torchvision.transforms as transforms
import pdb
import matplotlib.pyplot as plt
from models.resnet import *
# from advertorch.attacks import LinfPGDAttack


def visualize(image,fname):
    # torch tensor with scale zero to one
    image = image.detach().cpu().permute(1,2,0).numpy()
    # pdb.set_trace()
    plt.imsave('./figure/' + fname+'.png',image)
    return 


def distance_loss(X,X_adv):
    diff = (X - X_adv).reshape(-1)
    # distance  = torch.sqrt((diff**2).sum())/len(diff)
    distance = torch.abs(diff).sum()/len(diff)
    return distance


def color_reg(delta: torch.Tensor) -> torch.Tensor :
    return (delta.abs().mean([2,3]).norm() ** 2)/delta.shape[0]

def get_sim(delta : torch.Tensor) -> torch.Tensor :
    return ((delta[:,0] - delta[:,1])**2 + (delta[:,1] - delta[:,2])**2 + (delta[:,0] - delta[:,2])**2).norm(p = 2)/delta.shape[0]

def tv_loss(delta : torch.Tensor) -> torch.Tensor:
    x_wise = delta[:,:,:,1:] - delta[:,:,:,:-1]
    y_wise = delta[:,:,1:,:] - delta[:,:,:-1,:]
    return ((x_wise * x_wise).sum() + (y_wise + y_wise).sum())/delta.shape[0]


file_name = 'pgd_adversarial_training'
# file_name = 'basic_training'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
batch_size = 16

train_set, test_set = dataset.CIFAR10(normalize=False,download = False)
test_loader = torch.utils.data.DataLoader(test_set,batch_size = batch_size, shuffle=True)

net = ResNet18()

net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
checkpoint = torch.load('./checkpoint/' + file_name)
net.load_state_dict(checkpoint['net'])

# adversary = LinfPGDAttack(net, loss_fn=nn.CrossEntropyLoss(), eps=0.0314, nb_iter=7, eps_iter=0.00784, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
criterion = nn.CrossEntropyLoss()

eps = float(8/255)
alpha = float(2/255)
num_classes = 10


def test():
    print('\n[ Test Start ]')
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    my_adv_correct = 0
    total = 0
    noise_sd = 0.02
    gamma = 2
    steps = 7 

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        benign_loss += loss.item()

        _, predicted = outputs.max(1)
        benign_correct += predicted.eq(targets).sum().item()

        X_adv = inputs.detach()
        X_adv = X_adv + torch.zeros_like(X_adv).uniform_(-eps, eps)


        # PGD attack
        for i in range(steps):
            X_adv.requires_grad_()
            X_adv = X_adv.to(device)
           
            with torch.enable_grad():
                preds_adv = net(X_adv)
                loss = criterion(preds_adv,targets)
                # loss = softXEnt(preds_adv,targets)

            grad = torch.autograd.grad(loss, [X_adv])[0]
            X_adv = X_adv.detach() + alpha * torch.sign(grad.detach())
            # X_adv = X_adv.detach() + alpha *grad.detach()
            X_adv = torch.min(torch.max(X_adv, inputs - eps), inputs + eps)
            X_adv = torch.clamp(X_adv, 0, 1)


        preds_adv = net(X_adv)
        
        delta = torch.randn_like(X_adv,device = 'cuda',requires_grad= True)*noise_sd
        my_adv = inputs.detach()
        my_adv = my_adv + torch.zeros_like(my_adv).uniform_(-eps, eps)



        # Shadow attack
        for k in range(steps):
            
            preds = net(inputs + delta)
            distance = distance_loss(preds, preds_adv)
            loss = criterion(preds,targets) - 0.1*color_reg(delta) - 0.1*get_sim(delta) - 0.025 * tv_loss(delta) + 0.00 * distance_loss(preds, preds_adv)
            grad = torch.autograd.grad(loss, [delta])[0]
            
            delta = delta + alpha * torch.sign(grad)


        my_adv = torch.clip(inputs + delta,0,1)
        preds_my = net(my_adv)

        # visualize(inputs[0], "clean")
        # visualize(X_adv[0],"X_adv")
        # visualize(my_adv[0],"my_adv")

        adv_outputs = net(X_adv)
        my_outputs = net(my_adv)
        loss = criterion(adv_outputs, targets)
        adv_loss += loss.item()

        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()

        _, my_predicted = my_outputs.max(1)
        my_adv_correct += my_predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current my_adversarial test accuracy:', str(my_predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial test loss:', loss.item())

    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
    print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
    print('Total my_adversarial test Accuracy:',100. * my_adv_correct/total)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)

test()