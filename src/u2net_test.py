import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
import cv2
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from skimage.color import rgb2gray
import time
import onnxruntime
import yaml
import argparse
import logging
import warnings
import mlflow
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# normalize the predicted SOD probability map
sum_mae = 0
def normPRED(d):
    ma = np.max(d)
    mi = np.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def calc_mae(image_name, pred, mask_dir):
    global sum_mae
    predict = pred
    predict = predict.squeeze()
    img_name = image_name.split(os.sep)[-1]
    mask_name = img_name.split(".")[0] + ".png"
    mask = io.imread(os.path.join(mask_dir, mask_name))
    mask = (mask/255.).astype(np.float32)
    mask = cv2.resize(mask, predict.shape, interpolation=cv2.INTER_LANCZOS4)
    if np.ndim(mask) == 3:
        mask = rgb2gray(mask)
    sum_mae += mean_absolute_error(mask, predict)

def save_output(image_name,pred,d_dir):
    predict = pred.squeeze()
    im = Image.fromarray(predict*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    pb_np = np.array(imo)
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def eval(config):
    global sum_mae
    # --------- 1. parse config and get data ---------
    config = yaml.safe_load(open(config))
    resume = config['train']['model']['resume']
    device = config['train']['model']['device']
    checkpoint = config['train']['model']['save_path']
    model_name = config['train']['model']['version']
    action = config['evaluate']['evaluate_or_test']
    exp_name = config['train']['other']['exp_name']

    if action == "test":
        test_path = config['evaluate']['test_data']
        prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)

    if action == "evaluate":
        image_dir = os.path.join(os.getcwd(), 'DUTS-TE', 'DUTS-TE-Image')
        mask_dir = os.path.join(os.getcwd(), 'DUTS-TE', 'DUTS-TE-Mask')
    else:
        image_dir = os.path.join(os.getcwd(), test_path)

    img_name_list = glob.glob(image_dir + os.sep + '*')[:15]

    # --------- 2. dataloader ---------
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. inference session define ---------

    ort_session = onnxruntime.InferenceSession(checkpoint)

    i = 0
    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        logging.info("inference: {}".format(img_name_list[i_test].split(os.sep)[-1]))

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if device == "cuda":
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs_test)}
        start = time.time()
        ort_outs = ort_session.run(None, ort_inputs)

        # normalization
        pred = ort_outs[0][:,0,:,:]
        pred = normPRED(pred)
        end = time.time() - start
        logging.info("Inference exec time: {}".format(end))

        if action == "evaluate":
            calc_mae(img_name_list[i_test],pred, mask_dir)
            i+=1
            logging.info("ITERATIONS: {}".format(i))
            mean_mae = sum_mae / i
            logging.info("MEAN MAE MAIN: {}".format(mean_mae))

        # save results to test_results folder
        if action == "test":
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir, exist_ok=True)
            save_output(img_name_list[i_test],pred,prediction_dir)

    if action == "evaluate":
        client = MlflowClient()
        experiment = client.get_experiment_by_name("BG removal")
        query = 'tags.mlflow.runName = "{}"'.format(exp_name)
        runs = mlflow.search_runs(experiment.experiment_id, filter_string=query)
        client.log_metric(runs.iloc[0]['run_id'], "Mean Absolute Error on DUTS-TE", mean_mae)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    eval(config=args.config)
