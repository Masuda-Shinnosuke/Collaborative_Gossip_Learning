class LocalDataset(torch.utils.data.Dataset):
    def __init__(self,transform=None):
        self.transform = transform
        self.data=[]
        self.label=[]

    def __len__(self):
        return len(self.data)
    
    def __get__(self,index):
        out_data = self.data[index]
        out_label = self.label[index]
        if self.transform:
            out_data = self.transform(out_data)

        return out_data,out_label
    

class DatasetFromSubset(torch.utils.data.Dataset):
    def __init__(self,subset,transform=None):
        self.subeset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self,index):
        x,y = self.subeset[index]
        if self.transform:
            x = self.transform(x)
        return x,y
    
class GlobalDataset(torch.utils.data.Dataset):
  def __init__(self,federated_dataset,transform=None):
    self.transform = transform
    self.data = []
    self.label = []
    for dataset in federated_dataset:
      for (data,label) in dataset:
        self.data.append(data)
        self.label.append(label)

  def __getitem__(self, idx):
    out_data = self.data[idx]
    out_label = self.label[idx]
    if self.transform:
        out_data = self.transform(out_data)
    return out_data, out_label

  def __len__(self):
    return len(self.data)
