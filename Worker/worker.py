import configparser
import torch
config = configparser.ConfigParser()
config.read("config.ini")


class Worker():
    def __init__(self,trainset,valset,testset):
        self.trainLoader = torch.utils.data.DataLoader(trainset,batch_size=config["WORKER"]["batch_size"],shuffle=True,num_workers=2)
        self.valLoader = torch.utils.data.DataLoader(valset,batch_size=config["WORKER"]["batch_size"],shuffle=False,num_workers=2)
        self.trainLoader = torch.utils.data.DataLoader(testset,batch_size=config["WORKER"]["batch_size"],shuffle=False,num_workers=2)
        self.model=None
        self.train_data_num = len(trainset)
        self.test_data_num = len(testset)
        

    def local_train(self):
        acc_train,loss_train = train(self.model,args.criterion,self.trainloader,args.local_epochs)
        acc_valid,loss_valid = test(self.model,args.criterion,self.valloader)
        return acc_train,loss_train,acc_valid,loss_valid
    
def train(model,crterion,trainloader,epochs):
    optimizer = optim.SGD(model.parameters(),lr = args.lr,momentum = args.momentum,weight_decay = args.weight_decay)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        count = 0

        for (data,labels) in trainloader:
            data,labels = Variable(data),Variable(lables)
            data,lables = data.to(args.device),lables.to(args.device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs,lables)
            running_loss += loss.item()
            predicted = torch.argmax(outputs,dim=1)
            corecct += (predicted==lables).sum().item()
            count += len(labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
            optimizer.step()

    return 100.0*correct/count,running_loss/len(trainloader)

def test(model,criterion,testloader):
    model.eval()
    running_loss = 0
    correct = 0
    count = 0

    for (data,labels) in testloader:
        data,lables = data.to(args.device),labels.to(args.device)
        outputs = model(data)
        running_loss += criterion(outputs,lables).item()
        predicted = torch.argmax(outputs,dim = 1)
        correct += (predicted==lables).sum().item()
        coun += len(labels)

    accuracy = 100.0*correct/count
    loss = running_loss/len(testloader)

    return accuracy,loss