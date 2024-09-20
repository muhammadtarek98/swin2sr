import numpy as np
import torch
import cv2
import torchvision
import torchinfo
from models.network_swin2sr import Swin2SR
def load_model(device:torch.device,ckpt_path:str):
    upscale = 4
    window_size = 8
    height = (720 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = Swin2SR(upscale=8,in_chans=3, img_size=48, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle_aux', resi_connection='1conv')
    ckpt=torch.load(f=ckpt_path,map_location=device,weights_only=True)
    model.load_state_dict(ckpt)
    model=model.to(device=device)
    return model

def img_2_tensor(img):
    transform=torchvision.transforms.Compose(
        transforms=[
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size=(48,48),
                                          interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT),
            torchvision.transforms.Normalize(mean=[0,0,0],
                                             std=[1,1,1])])
    raw_image_tensor=transform(img)
    raw_image_tensor=torch.unsqueeze(input=raw_image_tensor, dim=0)
    return raw_image_tensor
def tensor_2_img(tensor:torch.Tensor):
    arr=np.clip(a=tensor,a_min=0,a_max=1)
    arr=(arr*255.0).astype(np.uint8)
    return arr

ckpt_path="model_zoo/swin2sr/Swin2SR_CompressedSR_X4_48.pth"
device=torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
model=load_model(device=device,
                 ckpt_path=ckpt_path)
model.eval()
image=cv2.cvtColor(src=
                   cv2.imread("/home/muahmmad/projects/Image_enhancement/dataset/underwater_imagenet/test/n02607072_10676.jpg"),
                   code=cv2.COLOR_BGR2RGB)
tensor=img_2_tensor(img=image)
with torch.no_grad():
    tensor=tensor.to(device)
    prediction=model(tensor)
    prediction=prediction[0]
print(prediction)
prediction=torch.squeeze(prediction)
prediction=torch.permute(input=prediction,dims=(1,2,0))
prediction=prediction.detach().cpu().numpy()
output_image=tensor_2_img(tensor=prediction)
cv2.imshow(winname="test",mat=output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
