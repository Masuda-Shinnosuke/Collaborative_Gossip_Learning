class Arguments():
    def __init__(self):
        self.batch_size = 40
        self.test_batch = 1000
        self.global_epochs = 500
        self.local_epochs = 2
        self.lr = None
        self.momentum = 0.9
        self.weight_decay = 10**-4.0
        self.clip = 20.0
        self.partience = 500
        self.worker_num = 20
        self.sample_num = 20
        self.unlabeleddata_size = 1000
        self.device = torch.device('cuda:0'if torch.cuda.is_available() else'cpu')
        self.criterion = nn.CrossEntropyLoss()
        
        self.alpha_label = 0.5
        self.alpha_size = 10
