import torch
import torch.nn.functional as F

class KeypointEncoder(object):
    def encode(self, keypoints, input_size, stride, hm_alpha, hm_sigma):
        '''
        For each image, create heatmaps and visibility masks. For 224*224 image and
        14 * 3 kps and 2 stride, return 112*112*14 heatmap
        Args:
            keypoints (tensor): [num_keypoints, 3] 
            input_size (int): original image size
            stride (int): Downsample multiplier
            hm_alpha (float):
                Alpha compositing for controlling the level of transparency.
            hm_sigma (float):
                Standard deviation for Gaussian function.
        Returns:
            img_heatmaps (tensor):
            img_vis_masks (tensor):
        '''
        num_keypoints = len(keypoints)
        # get heatmap size
        hm_size = input_size // stride
        # scale keypoints pos
        kpts = keypoints.clone()
        kpts[:, :2] = (kpts[:, :2] - 1.0) / stride
        #generate heatmap with zeros and vis_mask with zeros
        img_heatmaps = torch.zeros([num_keypoints, hm_size, hm_size])
        img_vis_masks = torch.zeros([num_keypoints])
        # add kps to img_heatmaps
        for i, kpt in enumerate(kpts):
            # For `visibility`, each digit represents the following:
            #
            # 1  = Visible
            # 0  = Not visible (i.e. occlude)
            # -1 = Does not exist (this kpt belongs to this category
            # but the annotation does not exist for this image)
            x, y, visibility = kpt

            if visibility >= 0:
                img_heatmaps[i] = self.__keypoint_to_heatmap(hm_size, x, y,
                                                             hm_alpha,
                                                             hm_sigma)
            img_vis_masks[i] = visibility

        return img_heatmaps, img_vis_masks
    def __keypoint_to_heatmap(self,hm_size,mu_x,mu_y,hm_alpha,hm_sigma):
        '''
        Args:
            hm_size (int):
                Size of the heatmap
            mu_x, mu_y (float):
                Means for Gaussian function.
            alpha (float):
                Alpha compositing for controlling the level of transparency.
            sigma (float):
                Standard deviation for Gaussian function.
        Returns:
            A (size, size) Gaussian heatmap.
        '''
        x = torch.linspace(0, hm_size - 1, steps=hm_size)
        y = x[:, None]
        # Generate the heatmap using the Gaussian function.
        Z = torch.exp(-(((x-mu_x)**2) / (2*hm_sigma**2) \
            + ((y-mu_y)**2) / (2*hm_sigma**2)))
        return hm_alpha * Z

if __name__=="__main__":
    kp_encode = KeypointEncoder()
    x = torch.randn(1,3,224,224)
    kp = torch.randn(14,3)
    img_heatmaps,vis_masks = kp_encode.encode(kp,224,2,2,0.11)

