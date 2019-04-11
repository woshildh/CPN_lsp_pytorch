from utils import Config
from argparse import ArgumentParser
from models import cpn_resnet50
import numpy as np
import cv2,torch

def transform(img,opt):
    img_h,img_w,_=img.shape
    #scale
    img = cv2.resize(img,(opt.img_size,opt.img_size))
    #change format
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # H*W*C --> C*H*W
    img[[0,2]] = img[[2,0]] # BGR --> RGB
    #normalize
    img = img / 255.0
    img = (img - np.reshape(np.asarray( opt.mu),(3,1,1))) / np.reshape(np.asarray(opt.sigma),(3,1,1))
    return torch.from_numpy(img)

def draw_kps(img,kps,opt):
    img_h,img_w,_ = img.shape
    # get image scale rate
    scale_x , scale_y = img_w/opt.img_size, img_h/opt.img_size
    # draw points
    for kp in kps:
        x = scale_x * kp[0] 
        y = scale_y * kp[1]
        cv2.circle(img,(int(x),int(y)),4,(0,255,0),-1)
    return img

def main():
    cfg = Config()
    parser = ArgumentParser(description="Please input parameters")
    parser.add_argument("--img_path")
    parser.add_argument("--weights_path")
    parser.add_argument("--save_path")
    args = parser.parse_args()
    # define model
    model = cpn_resnet50(cfg.num_kps)
    model.load_state_dict(torch.load(args.weights_path))
    # load img
    image = cv2.imread(args.img_path)
    img = transform(image,cfg.opt).unsqueeze(0).float()
    # get heatmaps
    out, p2 = model(img)
    # keypoints list
    kps = []
    for h in out[0]:
        pos = h.view(-1).argmax().item()
        x = (pos%h.size(1))*cfg.img_hp_rate
        y = (pos//h.size(1))*cfg.img_hp_rate
        kps.append([ x , y ])
    # draw points
    image = draw_kps(image,kps,cfg.opt)
    cv2.imwrite(args.save_path,image)

if __name__=="__main__":
    main()
