import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
import os,cv2
from sklearn.model_selection import train_test_split
from . import encode

__all__=["LSP"]

class LSP(Dataset):
    def __init__(self,mat_path,image_dir,phase,opt):
        self.mat_path=mat_path
        self.image_dir=image_dir
        self.load_data()
        self.phase = phase
        self.opt = opt
        self.mu = np.reshape(np.asarray(self.opt.mu),(3,1,1))
        self.sigma = np.reshape(np.asarray(self.opt.sigma),(3,1,1))
        self.encoder = encode.KeypointEncoder()
        if self.phase=="train":
            self.kp_data,_, self.image_list,_= train_test_split(self.kp_data,
                self.image_list,test_size=0.2,random_state=55)
        else:
            _,self.kp_data,_,self.image_list = train_test_split(self.kp_data,
                self.image_list,test_size=0.2,random_state=55)
    def __getitem__(self,index):
        img = cv2.imread(self.image_list[index]) #bgr
        kps = self.kp_data[index]
        img_h,img_w,_=img.shape
        # data augumentation
        if self.phase=="train":
            #flip
            random_flip = np.random.randint(0, 2)
            if random_flip:
                img = cv2.flip(img,1)
                kps[:,0] = img_w - kps[:,0]
            #rotation
            angle = np.random.randint(-30,30)
            M = cv2.getRotationMatrix2D(center=(img_w / 2, img_h / 2), angle=angle, scale=1)
            img = cv2.warpAffine(img, M, dsize=(img_w, img_h), flags=cv2.INTER_CUBIC)
            kps[:,2]=1
            kps[:,0:2] = np.matmul(kps, M.T)
        #scale
        scale = self.opt.img_size / max(img_h,img_w)
        img_h2, img_w2 = int(img_h * scale), int(img_w * scale)
        img = cv2.resize(img,(img_w2,img_h2))
        kps[:,0:2] *= scale
        #change format
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # H*W*C --> C*H*W
        img[[0,2]] = img[[2,0]] # BGR --> RGB
        #normalize
        img = img / 255.0
        img = (img - self.mu) / self.sigma
        return torch.from_numpy(img), torch.from_numpy(kps)
    def __len__(self):
        return len(self.image_list)
    def load_data(self):
        # load kepoints data
        self.kp_data = loadmat(self.mat_path)["joints"] #(3, 14, 2000)
        self.kp_data = np.transpose(self.kp_data,axes=(2,1,0)) #(2000,14,2)
        # generate image path list
        self.image_list = []
        for i in range(1,2001):
            im_name = "im{:04d}.jpg".format(i)
            self.image_list.append(os.path.join(self.image_dir,im_name))
    def vis_one_image(self,num=0):
        img_path = self.image_list[num]
        kp = self.kp_data[num]
        image = cv2.imread(img_path)
        for x,y,z in kp:
            cv2.circle(image,(int(x),int(y)),2,(0,0,255),-1)
        cv2.imwrite("./debug.jpg",image)
    def collate_fn(self,batch):
        imgs , kpts = zip(*batch)
        pad_imgs = torch.zeros(len(imgs), 3, self.opt.img_size, self.opt.img_size)
        heatmaps, vis_masks = [], []
        for i, img in enumerate(imgs):
            # If the image is smaller than `img_size`, we pad it with 0s.
            # This allows all images to have the same size.
            pad_imgs[i, :, :img.size(1), :img.size(2)] = img

            # For each image, create heatmaps and visibility masks.
            img_heatmaps, img_vis_masks = self.encoder.encode(kpts[i],
                                                              self.opt.img_size,
                                                              self.opt.hm_stride,
                                                              self.opt.hm_alpha,
                                                              self.opt.hm_sigma)

            # TODO: Can I avoid appending and do everything in torch?
            heatmaps.append(img_heatmaps)
            vis_masks.append(img_vis_masks)

        heatmaps = torch.stack(heatmaps)    # [batch_size, num_keypoints, h, w]
        vis_masks = torch.stack(vis_masks)  # [batch_size, num_keypoints]
        kpts = torch.stack(kpts)
        return pad_imgs, heatmaps, vis_masks,kpts

def test_lsp():
    class OPTION(object):
        def __init__(self,img_size,mu,sigma,hm_stride,hm_alpha,hm_sigma):
            self.img_size=img_size
            self.mu =mu
            self.sigma = sigma
            self.hm_stride=hm_stride
            self.hm_alpha=hm_alpha
            self.hm_sigma=hm_sigma
    opt=OPTION(480,(0.42,0.415,0.443),(0.225,0.210,0.231),2,0.3,0.1)
    lsp =LSP("/home/lenstr/ldh/datasets/lsp/joints.mat",
        "/home/lenstr/ldh/datasets/lsp/images/","train",opt)
    from torch.utils.data import DataLoader
    loader = DataLoader(lsp,4,False,collate_fn=lsp.collate_fn)
    for img,hp,masks,kpts in loader:
        print(img.size(),hp.size(),masks.size(),kpts.size())

if __name__=="__main__":
    test_lsp()