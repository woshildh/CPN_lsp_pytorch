import os

class Option(object):
    def __init__(self,img_size,mu,sigma,hm_stride,hm_alpha,hm_sigma):
        self.img_size=img_size
        self.mu =mu
        self.sigma = sigma
        self.hm_stride=hm_stride
        self.hm_alpha=hm_alpha
        self.hm_sigma=hm_sigma

class Config(object):
    # model params
    num_kps = 14
    pretrained = "./logs/resnet50.pth"
    # data params
    lsp_mat = "/home/lenstr/ldh/datasets/lsp/joints.mat"
    lsp_images = "/home/lenstr/ldh/datasets/lsp/images/"
    mu = [0.485, 0.456, 0.406]
    sigma = [0.229, 0.224, 0.225]
    hm_stride = 4
    hm_alpha = 100
    img_size = 224
    hm_sigma = img_size / hm_stride / 16.
    opt = Option(img_size,mu,sigma,hm_stride,hm_alpha,hm_sigma)
    # train params
    use_gpu = True
    base_lr = 0.001
    epochs = 100
    stones = [10,30,50,80] # lr * 0.1 when epoch==stones[i]
    batch_size = 16
    num_workers = 4
    tb_dir = "./logs/tblog/"
    weights_dir="./logs/weights/"
    weights_name="cpn_lsp_without_ohkm"
    # demo params
    weights_path = "./"
    img_hp_rate = 4
