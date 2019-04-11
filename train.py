import models
import loss
import datasets
import utils
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

def main():
    # read params
    cfg = utils.Config()
    # define dataloader
    trainset = datasets.LSP(cfg.lsp_mat,cfg.lsp_images,"train",cfg.opt)
    trainloader = DataLoader(trainset,cfg.batch_size,
        True,num_workers=cfg.num_workers,collate_fn=trainset.collate_fn)
    # define model
    model = models.cpn_resnet50(cfg.num_kps,pretrained=cfg.pretrained)
    if cfg.use_gpu:
        model = model.cuda()
    # define logger,checkpoint,avgmeter
    logger = utils.Logger(cfg.tb_dir)
    checkpoint = utils.Checkpoint(cfg.weights_dir,cfg.weights_name,False,3)
    #define optimizer
    optimizer = optim.SGD(model.parameters(),lr=cfg.base_lr,
        nesterov=True,momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,cfg.stones,gamma=0.1)
    # define criterion
    criterion = loss.CPNLoss(num_kps=cfg.num_kps)
    print("There are {} images".format(len(trainset)))
    for epoch in range(1,cfg.epochs+1):
        all_loss,global_loss,refine_loss = train(model,optimizer,
            trainloader,criterion,epoch,cfg.use_gpu)
        logger.log(step=epoch,content={"all_loss":all_loss,"global_loss":
            global_loss,"refine_loss":refine_loss})
        checkpoint.save(model.state_dict(),epoch,performence=all_loss)
    print("train end ...")

def train(model,optimizer,dataloader,criterion,epoch,use_gpu):
    global_loss_am = utils.AverageMeter()
    refine_loss_am = utils.AverageMeter()
    all_loss_am = utils.AverageMeter()
    for step,data in enumerate(dataloader):
        img,hp,masks,kpts = data
        if use_gpu:
            img,hp,masks,kpts = img.cuda(),hp.cuda(),masks.cuda(),kpts.cuda()
        out , p2 = model(img)
        all_loss, global_loss , refine_loss = criterion(p2,out,hp,masks)
        # optimize model
        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()
        #log loss
        global_loss_am.update(global_loss.item())
        refine_loss_am.update(refine_loss.item())
        all_loss_am.update(all_loss.item())
        #print info
        print("{} peoch, {} step, all loss is {:.4f}, global loss is {:.4f}, refine loss is {:.4f}".format(
            epoch,step,all_loss.item() , global_loss.item(), refine_loss.item()
        ))
    return all_loss_am.avg, global_loss_am.avg , refine_loss_am.avg

if __name__=="__main__":
    main()
