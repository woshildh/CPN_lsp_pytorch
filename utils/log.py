import tensorboardX as tb
import torch,os

class Logger(object):
    def __init__(self,tb_path="./logs/tblog/"):
        self.writer = tb.SummaryWriter(log_dir=tb_path)
    def log(self,step=1,content={}):
        for key,value in content.items():
            self.writer.add_scalar(key,value,step)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / (1e-6 + self.count)

class Checkpoint(object):
    def __init__(self,root_dir,name,large=False,save_most=5):
        self.name = os.path.join(root_dir,name+"_{}.pth")
        self.large = large
        self.save_most = save_most
        self.best_performence = -100000 if large else 1000000
    def save(self,state_dict,epoch,performence=0):
        # save weights
        torch.save(state_dict,self.name.format(epoch))
        # check and delete over-time weights
        if os.path.exists(self.name.format(epoch-self.save_most)):
            os.remove(self.name.format(epoch-self.save_most))
        # save best weights if performence is best
        flag = False
        if self.large and performence>self.best_performence:
            flag=True; self.best_performence=performence;
        if not self.large and performence<self.best_performence:
            flag =True; self.best_performence=performence;
        if flag:
            torch.save(state_dict,self.name.format("best"))

