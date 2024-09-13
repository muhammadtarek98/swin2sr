from models.network_swin2sr import Swin2SR as net
import torch,torchvision,torchinfo,cv2
import numpy as np
def load_model(device:torch.device,
               model:torch.nn.Module,
               ckpt_dir:str="/home/muahmmad/projects/Image_enhancement/waternet/weights/waternet_exported_state_dict-daa0ee.pt"):

    ckpt=torch.load(f=ckpt_dir,map_location=device,weights_only=True)
    print(ckpt.keys())
    model.load_state_dict(state_dict=ckpt)
    model=model.to(device=device)
    return model
def transform_array_to_image(arr):
    arr=np.clip(a=arr,a_min=0,a_max=1)
    arr=(arr*255.0).astype(np.uint8)
    return arr
def transform_image(img,single_image:bool=True):
    trans=torchvision.transforms.Compose(transforms=[
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=(480,480),interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT),
        #torchvision.transforms.Normalize(mean=[0,0,0],
        #                                 std=[1,1,1])
    ])
    raw_image_tensor = trans(img)
    if single_image:
        return {"X":torch.unsqueeze(input=raw_image_tensor,dim=0)}
    else:
        wb_tensor=trans(wb)
        gc_tensor=trans(gc)
        he_tensor=trans(he)
        return {"X":torch.unsqueeze(input=raw_image_tensor,dim=0),
                "wb":torch.unsqueeze(input=wb_tensor,dim=0),
                "gc":torch.unsqueeze(input=gc_tensor,dim=0),
                "he":torch.unsqueeze(input=he_tensor,dim=0)}

if __name__ == '__main__':
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt="/home/muahmmad/projects/Image_enhancement/swin2sr/model_zoo/swin2sr/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth"
    # compressed_sr
    """model = net(upscale=4, in_chans=3, img_size=48, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle_aux', resi_connection='1conv')
    """
    #Real_SR
    model=net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
    #print(model)
    model=load_model(device=device,ckpt_dir=ckpt,model=model)
    image=cv2.imread(filename="/home/muahmmad/projects/Image_enhancement/waternet/264_cam1left_2016_10_21_07_10_15.jpg",
                     )
    #image=cv2.cvtColor(src=image,code=cv2.COLOR_BGR2RGB)

    raw_image_tensor=transform_image(img=image)["X"]
    with torch.no_grad():
        raw_image_tensor = raw_image_tensor.to(device=device)
        pred=model(raw_image_tensor)
    print(len(pred))
    pred=pred.squeeze_()
    pred = torch.permute(input=pred, dims=(1, 2, 0))
    output = pred.detach().cpu().numpy()
    output=transform_array_to_image(output)
    output=cv2.resize(src=output,dsize=(720,720))
    #cv2.imshow(winname="test",mat=output)
    cv2.imwrite(filename="264_cam1left_2016_10_21_07_10_15.jpg",img=output)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()


