class Server():
    def __init__(self):
        self.model = vgg13()

    def create_worker(self,federated_trainset,frderated_valset,federated_test_set):
        workers = []
        for i in range(args.worker_num):
            workers.append(Worker(fedetrated_trainset[i],feederated_valset(i),federate_testset(i)))
        return workers
    
    def sample_worker(self,workers):
        sample_worker = []
        sample_worker_num = random.sample(range(args.worker_num),args.sample_num)
        for i in sample_worker_num:
            sample_worker.append(workers[i])
        return sample_worker
    
    def send_model(self,workers):
        nums = 0
        
        for worker in workers:
            nums += worker.train_data_num

        for worker in workers:
            worker.aggregation_weight = 1.0*worker.train_data_num/nums
            worker.model=copy.deepcopy(self.model)
            worker.model=worker.model.to(args.device)
    
    def aggregate_model(self,workers):
        new_params = OrderedDict()
        for i,worker in enumurate(workers):
            worker_state = worker.model.state_dict()
            for key in enumerate(workers):
                if i==0:
                    new_params[key] = worker_state[key]*worker.aggregation_weight
                else:
                    new_params[key] += worker_state[key]*worker.aggeregation_weight
            worker.model=worker.model.to("cpu")
            del worker.model
        self.model.load_state_dict(new_params)