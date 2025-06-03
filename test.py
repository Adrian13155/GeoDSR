import sys
sys.path.append("/home/cjj/projects/Depth_SR")
import argparse
import torch
import numpy as np
from models import *
import cv2
import logging
from torchvision.transforms import transforms
from datasets import *

def calc_rmse(a, b, minmax):
    """
    NYU_Depth_V2_Dataset
    参考:https://github.com/yanzq95/SGNet/blob/main/utils.py
    """
    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]
    
    a = a*(minmax[0]-minmax[1]) + minmax[1]
    b = b*(minmax[0]-minmax[1]) + minmax[1]
    a = a * 100
    b = b * 100
    
    return torch.sqrt(torch.mean(torch.pow(a-b,2)))

def prepare_data(data):
    if isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, np.ndarray):
                data[i] = torch.from_numpy(v).cuda()
            if torch.is_tensor(v):
                data[i] = v.cuda()
    # 基本可以判断，dataloader输出的是dict，而且是大的dict  
    elif isinstance(data, dict):
        for k, v in data.items():
            # print(k)
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v).cuda()
            if torch.is_tensor(v):
                data[k] = v.cuda()
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data).cuda()
    else: # is_tensor
        data = data.cuda()

    return data

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def test(opt):
    logger = get_logger(f"{opt.logPath}/result_{opt.other_information}{opt.scale}x.log")
    logger.info(opt)
    net = GASA_model(opt, 2, 1, 4).cuda()
    checkpoint_dict = torch.load(opt.checkpointPath)
    net.load_state_dict(checkpoint_dict['model'])

    data_transform = transforms.Compose([transforms.ToTensor()])

    dataset_name = "NYU"
    dataset = None
    if dataset_name == 'NYU':
        test_minmax = np.load('%s/test_minmax.npy' % opt.root_dir)
        dataset = NYU_v2_datsetForGeoDSR(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=False)
        rmse = np.zeros(449)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    data_num = len(dataloader)

    with torch.no_grad():
        net.eval()
        if dataset_name == 'NYU':
            for idx, data in enumerate(dataloader):
                data = prepare_data(data)
                img_predict= net(data)
                gt = data['hr_depth']
                minmax = test_minmax[:, idx]
                minmax = torch.from_numpy(minmax).cuda()
                rmse[idx] = calc_rmse(img_predict[0,0], gt[0, 0], minmax)
                
                path_save_pred = '{}/{}.png'.format(opt.results_dir, idx + 1)
                
                # Save results  (Save the output depth map)
                # pred = img_predict[0,0] * (minmax[0] - minmax[1]) + minmax[1]
                # pred = pred * 1000.0
                # pred = pred.cpu().detach().numpy()
                # pred = pred.astype(np.uint16)
                # pred = Image.fromarray(pred)
                # pred.save(path_save_pred)
                
                # visualization  (Visual depth map)
                pred = img_predict[0, 0]
                pred = pred.cpu().detach().numpy()
                cv2.imwrite(path_save_pred, pred * 255.0)   
                
                logger.info(f"idx = {idx + 1}, rmse[{idx + 1}] = {rmse[idx]}")
            logger.info(f"rmse.mean: {rmse.mean()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=16, help='scale factor')
    parser.add_argument('--device', default="0", type=str, help='which gpu use')
    parser.add_argument("--root_dir", type=str, default='/data/cjj/dataset/NYU_V2', help="root dir of dataset")
    parser.add_argument("--checkpointPath",type=str, default="/home/cjj/projects/Depth_SR/compare_methods/GeoDSR/checkpoints/GASA_best.pth.tar")
    parser.add_argument("--other_information",type=str, default="GeoDSR")
    parser.add_argument("--results_dir",type=str, default="/home/cjj/projects/Depth_SR/compare_results/result_GeoDSR/16x")
    parser.add_argument("--logPath",type=str, default="/home/cjj/projects/Depth_SR/compare_methods/GeoDSR/result")

    opt = parser.parse_args()

    
    test(opt)