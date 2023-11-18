from medpy import metric
from scipy.ndimage import zoom
import torchvision.transforms
import torch.optim
from load_data import ValGenerator, ImageToImage2D_kfold
from torch.utils.data import DataLoader
import warnings
import time
warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
from nets.UNet import *
from nets.UDTransNet import UDTransNet
from nets.TF_configs import get_model_config
import os
from utils import *
import cv2
from PIL import Image

def vis_save_synapse(input_img, pred, mask, save_path):
    blue   = [30,144,255] # aorta
    green  = [0,255,0]    # gallbladder
    red    = [255,0,0]    # left kidney
    cyan   = [0,255,255]  # right kidney
    pink   = [255,0,255]  # liver
    yellow = [255,255,0]  # pancreas
    purple = [128,0,255]  # spleen
    orange = [255,128,0]  # stomach
    if pred is not None:
        pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2RGB)
        input_img = np.where(pred==1, np.full_like(input_img, [0,0,255]   ), input_img)
        input_img = np.where(pred==2, np.full_like(input_img, [0,255,255]), input_img)
        input_img = np.where(pred==3, np.full_like(input_img, green ), input_img)
        input_img = np.where(pred==4, np.full_like(input_img, cyan  ), input_img)
        input_img = np.where(pred==5, np.full_like(input_img, pink  ), input_img)
        input_img = np.where(pred==6, np.full_like(input_img, yellow), input_img)
        input_img = np.where(pred==7, np.full_like(input_img, purple), input_img)
        input_img = np.where(pred==8, np.full_like(input_img, orange), input_img)
    else:            
        input_img = np.where(mask==1, np.full_like(input_img, [0,0,255]  ), input_img)
        input_img = np.where(mask==2, np.full_like(input_img, [0,255,255] ), input_img)
        input_img = np.where(mask==3, np.full_like(input_img, green   ), input_img)
        input_img = np.where(mask==4, np.full_like(input_img, cyan  ), input_img)
        input_img = np.where(mask==5, np.full_like(input_img, pink  ), input_img)
        input_img = np.where(mask==6, np.full_like(input_img, yellow), input_img)
        input_img = np.where(mask==7, np.full_like(input_img, purple), input_img)
        input_img = np.where(mask==8, np.full_like(input_img, orange), input_img)
    cv2.imwrite(save_path, input_img)
    print(save_path)
if __name__ == '__main__':
    ## PARAMS
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    ensemble_models=[]
    test_session = config.test_session

    for i in range(0,1):
        if config.task_name == "GlaS":
            test_num = 17
            model_type = config.model_name
            # model_path = "./GlaS_kfold/"+model_type+"/"+test_session+"/models/fold_"+str(i+1)+"/best_model-"+model_type+".pth.tar"
            model_path = "/root/UDTransNet/GlaS_kfold/UDTransNet/Test_session_08.16_09h16/models/fold_3/best_model-UDTransNet.pth.tar"
        elif config.task_name == "Synapse":
            test_num = 12
            model_type = config.model_name
            model_path = "./Synapse_kfold/"+model_type+"/"+test_session+"/models/fold_"+str(i+1)+"/best_model-"+model_type+".pth.tar"

        maxi = 5
        if not os.path.exists(model_path):
            maxi = i
            print("====",maxi, "models loaded ====")
            break
        checkpoint = torch.load(model_path, map_location='cuda')

        if model_type == 'UNet':
            model = UNet(n_channels=config.n_channels,n_classes=config.n_labels)
        elif model_type == 'R34_UNet':
            model = R34_UNet(n_channels=config.n_channels,n_classes=config.n_labels)
        elif model_type == 'UDTransNet':
            config_vit = get_model_config()
            model = UDTransNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels, img_size=config.img_size)

        else: raise TypeError('Please enter a valid name for the model type')

        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
            model = nn.DataParallel(model, device_ids=[0,1])

        model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded !')
        # model.eval()
        ensemble_models.append(model)
    orderlists = os.listdir(config.test_dataset)
    orderlists.sort()
    print(orderlists)
    for i in range(0, len(orderlists)):
        filepath=os.path.join(config.test_dataset,orderlists[i]+"/originpic")
        filelists = os.listdir(filepath)
        print(filepath)
        test_num = len(filelists)
        tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
        test_dataset = ImageToImage2D_kfold(filepath,
                                            tf_test,
                                            image_size=config.img_size,
                                            task_name=config.task_name,
                                            filelists=filelists,
                                            split='test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


        end = time.time()
        with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
            for i, (sampled_batch, names) in enumerate(test_loader, 1):
                test_data = sampled_batch['image']
                att_vis_path = filepath.replace("originpic", "udmask/")
                # att_vis_path = config.test_dataset + 'udmask/'
                # print(att_vis_path)
                if not os.path.exists(att_vis_path):
                    os.makedirs(att_vis_path)
                num = test_data.size(0)
                for idx in range(num):
                    res_vis = []
                    input_512 = torchvision.transforms.functional.to_pil_image(test_data[idx])
                    input_vis = input_512
                    input_vis = input_vis.resize((512, 512))
                    input = torchvision.transforms.functional.to_tensor(input_vis).unsqueeze(0)     # 增加一维
                    for model_ in ensemble_models:
                        output = model_(input.cuda())
                        predict_save = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
                        predict_save = predict_save.cpu().data.numpy()
                        res_vis.append(output)
                    res_vis = torch.cat(res_vis,dim=0)
                    predict_save = torch.argmax(torch.softmax(res_vis.mean(0), dim=0), dim=0).cpu().data.numpy().astype(np.uint8)
                    vis_save_synapse(input_512, predict_save, None, save_path = att_vis_path + names[0])
                
                torch.cuda.empty_cache()
                pbar.update()




