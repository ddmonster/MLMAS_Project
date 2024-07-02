from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import cv2

from scipy.signal import find_peaks
# GET MATRIX
def get_tfuse_matrix():
    src = np.float32([
        (300, 512), (720, 512),
        (400, 350),(600, 350)
    ])
    dst = np.float32([
        (300, 512), (720, 512),
        (300, 300),(720, 300)
    ])
    Matrix = cv2.getPerspectiveTransform(src, dst)
    return Matrix

def get_lane_pos(arr):
    length = len(arr)
    center = 1024 / 2
    print(center)
    left = [i for i in arr if i <center]
    right = [i for i in arr if i> center]
    if len(left) == 0 :
        left = -1
    else:
        left = center - sum(left)/len(left) 
    if len(right) == 0:
        right = -1
    else:
        right = sum(right)/len(right) - center
    return left,right


class LaneDetection:
    def __init__(self,model_path="./best_model_multi_dice_loss.pth") -> None:
        self.model = torch.load(model_path)
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn('efficientnet-b0', 'imagenet')
        self.DEVICE  = "cuda"
    def resize_width_crop_height(self, img, target_size = (1024,512)):
        """
        Resize the image based on the width and crop the height to the target size.
        
        Parameters:
        - img: PIL Image object
        - target_size: tuple of (width, height)
        
        Returns:
        - img: PIL Image object of the target size
        """
        target_width, target_height = target_size
        
        # Calculate the new height maintaining the aspect ratio
        aspect_ratio = img.width / img.height
        new_height = int(target_width / aspect_ratio)
        
        # Resize the image based on the width
        img = img.resize((target_width, new_height), Image.ANTIALIAS)
        
        # If the new height is greater than the target height, crop the image from the center
        if new_height > target_height:
            top = (new_height - target_height) / 2
            bottom = (new_height + target_height) / 2
            img = img.crop((0, top, target_width, bottom))
        
        return img
    
    def predict(self,img,skip=False):
        if not skip:
            image = Image.fromarray(img, 'RGBA')

            #resize and corp to 512X1024
            img_resized = self.resize_width_crop_height(image)
                    # Convert RGBA to RGB
            img_resized = img_resized.convert('RGB')

            # covert to numpy array
            img_array = np.array(img_resized)
        else:
            img_array = img

        preprocessed = self.preprocessing_fn(img_array)

        # trans (width,height,channel) to (channel,width,height). change type to float32
        img_array = preprocessed.transpose(2,0,1).astype('float32')


        x_tensor = torch.from_numpy(img_array).to(self.DEVICE).unsqueeze(0)

        #predict result overall img rs[0,0,:,:], left line rs[0,0,:,:], right line rs[0,0,:,:]
        rs = self.model.predict(x_tensor)
        mask = rs[0,0,:,:].cpu().numpy().copy()
        left = rs[0,1,:,:].cpu().numpy().copy()
        right = rs[0,2,:,:].cpu().numpy().copy()
        return mask, left , right,preprocessed,rs
    
    def post_processing(self,output,matrix=get_tfuse_matrix()):
        o= output
        warped_mask = cv2.warpPerspective(o, matrix, (1024, 512),cv2.INTER_LINEAR)

        vertical_projection = np.sum(np.where(warped_mask >= 0.1, 0, 1), axis=0)

        peaks, _ = find_peaks(vertical_projection, height=0)
        fpeaks = [index for index in peaks if vertical_projection[index] > 17]
        fpeaks = [index for index in peaks if vertical_projection[index] > 10]

        print(f"==>> fpeaks: {fpeaks}")
        if len(fpeaks) == 0:
            return get_lane_pos([])
        filtered_peaks = [fpeaks[0]]
        for i in range(1, len(fpeaks)):
            if fpeaks[i] - filtered_peaks[-1] < 10:
                if fpeaks[i] > filtered_peaks[-1]:
                    filtered_peaks[-1] = fpeaks[i]
            else:
                filtered_peaks.append(fpeaks[i])
        print(f"==>>filtered  fpeaks: {filtered_peaks}")
        return get_lane_pos(filtered_peaks)
    def get_pos(self,img):
        o,left,right,pr,rs=self.predict(img)
        return self.post_processing(o)

if __name__ =="__main__":
    from pathlib import Path
    model = LaneDetection()
    Tfuse = Path("/home/ddmonster/MLMAS_Project/imgdata/tfuse")
    tfuse =list(Tfuse.glob("*.png"))
    img = Image.open(tfuse[0])

    print(model.get_pos(np.array(img)))