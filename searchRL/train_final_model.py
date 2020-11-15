import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#from model_torch import Net
import torch.nn as nn
import neptune
import torch.nn.functional as F
from model_torch import model_fn
from get_flops import find_flops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)


#device = "cpu"
class Final_model:
    '''
    Get final accuracy of the trained model
    '''
    def __init__(self, epochs=5, child_batchsize=128,actions=None):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.

        Args:
            dataset: a tuple of 4 arrays (X_train, y_train, X_val, y_val)
            epochs: number of epochs to train the subnetworks
            child_batchsize: batchsize of training the subnetworks
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batchsize = child_batchsize
        
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batchsize,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batchsize,
                                                 shuffle=False, num_workers=2)
        
        self.net = model_fn(actions,10).to(self.device)
        self.macs,_=find_flops(model_fn(actions,10) , input = -1)
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        
    def get_accuracy(self):
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.Adam(self.net.parameters(), lr=0.001,betas=(0.9, 0.999))
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs.to(self.device))
                loss = criterion(outputs, labels.to(self.device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in self.testloader:
                    images, labels = data
                    outputs = self.net(images.to(self.device))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted.cpu() == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
        
        acc=correct/total
        return acc
    
if __name__=='__main__':
    ##### Best states by NAS RL with accuracy as reward ######
    actions=[3,32,3,32,3,64,3,64]
    final_model=Final_model(epochs=70, child_batchsize=128,actions=actions)
    print("Final accuracy of the models",final_model.get_accuracy())
    print("Flops of the model",final_model.macs)
    
    
    
    
    #### Best states across N trials for NAS without training
#     actions_list=[[3, 32, 3, 16, 3, 32, 1, 32], [3, 32, 1, 64, 3, 16, 3, 32], [3, 32, 3, 16, 3, 16, 1, 32], [3, 64, 1, 32, 3, 64, 3, 32], [3, 32, 3, 16, 3, 64, 3, 16], [3, 16, 3, 64, 1, 64, 1, 32], [3, 16, 3, 16, 1, 32, 3, 16], [3, 16, 3, 16, 1, 32, 1, 32], [3, 64, 1, 32, 3, 32, 3, 32], [3, 64, 1, 32, 3, 16, 1, 32]]
    
#     list_accuracies=[]
#     list_flops=[]
#     for actions in actions_list:
#         final_model=Final_model(epochs=70, child_batchsize=128,actions=actions)
#         list_accuracies.append(final_model.get_accuracy())
#         list_flops.append(final_model.macs)
#     print("Final accuracy of the model",np.mean(list_accuracies))
#     print("Flops of the model",np.mean(list_flops))
    
    
    