import os
import sys
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from model3d import resnet

from resnet import resnet50_multi_new,resnet50_multi_new_cbma,resnet50bam,resnet50se,resnet50_multi_new2


import numpy as np


import matplotlib.pyplot as plt

from torch.nn import DataParallel

import torch.nn.init as nn_init

from os import path
from PIL import Image
import numpy as np
import pandas as pd


import datetime
import time
from utils_loss import FocalLoss


from dataloader.image_transforms import Image_Transforms

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 计算回归指标的函数
def compute_regression_metrics(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    corr = r2_score(targets, predictions)
    return mse, mae, rmse, corr

def writefile(name, list):
    # print(list)

    f = open(name+'.txt', mode='w')  # 打开文件，若文件不存在系统自动创建。
    # f.write(Loss_list)  # write 写入
    for i in range(len(list)):
        s = str(list[i]).replace('{', '').replace('}', '').replace("'", '').replace(':', ',') + '\n'
        f.write(s)
    f.close()



def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print(path + ' 目录已存在')
        return False


#这个程序要重新改，改成从csv分别读取test和train信息







path_join = path.join


#单一样本
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, im_dir, im_names, im_labels, im_path,im_transforms=None):
        self.im_dir = im_dir
        self.im_labels = im_labels
        self.im_names = im_names
        self.im_path_head=im_path
        if im_transforms:
            self.im_transforms = im_transforms
        else:
            self.im_transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):

        return len(self.im_labels)

    def __getitem__(self, idx):




        im_file = os.path.join(self.im_dir,self.im_path_head[idx],'眼底照相',
                               self.im_names[idx].split(';')[0])
        im = Image.open(im_file).convert('RGB')

        im = self.im_transforms(im)

        return im, self.im_labels[idx], self.im_path_head[idx]


        # return im, self.non_im_inputs[idx], self.im_labels[idx]
 
 

def load_data(label_path, train_lists, img_path,
              batchsize, im_transforms,type):   


    train_sets = []
    train_loaders = []




    for train_list in train_lists:
        full_path_list = path_join(label_path,train_list)
        df = pd.read_csv(full_path_list)
        im_names = df['YD_PATH'].to_numpy()

        # im_labels = torch.tensor(df[classes].to_numpy(), dtype=torch.float)
        im_labels=df['XIANRAN_SL'].to_numpy()
        im_path=df['PATH'].to_numpy()


    
        train_sets.append(CustomDataset(img_path, im_names, im_labels , im_path,im_transforms))
        train_loaders.append(torch.utils.data.DataLoader(train_sets[-1], batch_size=batchsize, shuffle=True,num_workers=8))
        print('Size for {0} = {1}'.format(train_list, len(im_names)))

        #这里的问题，因为原来的数据分了很多歌批次，不同的csv。这个就一个csv。原来是trainloader 数组拼出来。现在就取0就好了

    return train_loaders[0]





#插帧算法


import random


def get_random_images(image_files, max_img, seed=123):
    total_frames = len(image_files)
    if total_frames <= max_img:
        if seed is not None:
            random.seed(seed)
        output = image_files.copy()
        if seed is not None:
            random.shuffle(output)
        while len(output) < max_img:
            output += image_files
        return sorted(output[:max_img])
    
    if seed is not None:
        random.seed(seed)

    indices = random.sample(range(total_frames), max_img)
    indices.sort()
    image_list = [image_files[i] for i in indices]
    return image_list






#批量样本
class CustomDataset_Multi(torch.utils.data.Dataset):
    def __init__(self, im_dir, im_names, im_labels, im_path,im_transforms=None):
        self.im_dir = im_dir
        self.im_labels = im_labels
        self.im_names = im_names
        self.im_path_head=im_path
        if im_transforms:
            self.im_transforms = im_transforms
        else:
            self.im_transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):

        return len(self.im_labels)

    def __getitem__(self, idx):

        # seed = 123


        image_list=get_random_images( self.im_names[idx].split(';'),24)

        modified_list = [os.path.join(self.im_dir,self.im_path_head[idx],'OCT', string) for string in image_list]
            

        images = []

        for image_path in modified_list:



            try:
                im = Image.open(image_path).convert('RGB')
                im = self.im_transforms(im)
                images.append(im)
            except:
                print('Error: Failed to open or verify image file {}'.format(image_path))

        clip=images
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        # return images, self.im_labels[idx], self.im_path_head[idx]
        return clip, self.im_labels[idx], self.im_path_head[idx]


        # return im, self.non_im_inputs[idx], self.im_labels[idx]
 
 

def load_data_multi(label_path, train_lists, img_path,
              batchsize, im_transforms,type):   


    train_sets = []
    train_loaders = []




    for train_list in train_lists:
        full_path_list = path_join(label_path,train_list)
        df = pd.read_csv(full_path_list)
        im_names = df['SP_PATH'].to_numpy()

        # im_labels = torch.tensor(df[classes].to_numpy(), dtype=torch.float)
        im_labels=df['XIANRAN_SL_new'].to_numpy()
        im_path=df['PATH'].to_numpy()


    
        train_sets.append(CustomDataset_Multi(img_path, im_names, im_labels , im_path,im_transforms))
        train_loaders.append(torch.utils.data.DataLoader(train_sets[-1], batch_size=batchsize, shuffle=True,num_workers=8))
        print('Size for {0} = {1}'.format(train_list, len(im_names)))

        #这里的问题，因为原来的数据分了很多歌批次，不同的csv。这个就一个csv。原来是trainloader 数组拼出来。现在就取0就好了

    return train_loaders[0]












def main():

    taskname='_9_hbspoct_3d'

    modeltype="3dresnet50"
# from resnet import resnet50_multi_new,resnet50_multi_new_cbma,resnet50bam,resnet50se  3dresnet50


    label_path='/root/work2023/deep-learning-for-image-processing-master/data_set/TRSL_ALL'



    path='/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_sl/'+taskname

    IMAGE_PATH = '/vepfs/gaowh/tr_eyesl/'

    TRAIN_LISTS = ['train.csv']
    TEST_LISTS = ['test.csv']






    mkdir(path)
    save_path_best = path+'/resNet50'+taskname+'_best.pth'
    save_path_final = path+'/resNet50'+taskname+'_final.pth'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("using {} device.".format(device))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()





    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path




    batch_size = 16

    # 4 load dataset
    train_transforms = Image_Transforms(mode='train', dataloader_id=1, square_size=256, crop_size=224).get_transforms()
    val_transforms = Image_Transforms(mode='val', dataloader_id=1, square_size=256, crop_size=224).get_transforms()


    # Create training and test loaders
    validate_loader = load_data_multi(label_path, TEST_LISTS, IMAGE_PATH, batch_size, val_transforms,'test')

    train_loader = load_data_multi(label_path, TRAIN_LISTS, IMAGE_PATH, batch_size, val_transforms,'train')



















    dfres = pd.read_csv(path_join(label_path, TEST_LISTS[0]))
    val_num=dfres.shape[0]

    dftrain = pd.read_csv(path_join(label_path, TRAIN_LISTS[0]))

    train_num=dftrain.shape[0]




















    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))







    if modeltype == "resnet50_multi_new":
        net = resnet50_multi_new()
    elif modeltype == "resnet50_multi_new_cbma":
        net = resnet50_multi_new_cbma()
    elif modeltype == "resnet50bam":
        net = resnet50bam()
    elif modeltype == "resnet50se":
        net = resnet50se()
    elif modeltype == "resnet50_multi_new2":
        net = resnet50_multi_new2()

    elif modeltype == "3dresnet50":
        net = resnet.generate_model(model_depth=50,n_classes=1)
        
    else:
        raise ValueError("Invalid modeltype")

    

    print(net)






    if modeltype != "3dresnet50":

        # 加载预训练参数
        predict_model = torch.load("/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_sl/resnet50-pre.pth")
        state_dict = {k: v for k, v in predict_model.items() if k in net.state_dict().keys()}  # 寻找网络中公共层，并保留预训练参数

        # 获取预训练参数中的conv1.weight
        pretrained_weight = state_dict['conv1.weight']

        # 创建新的conv1.weight，将预训练参数中的权重复制到新的权重中
        new_weight = torch.zeros((64, 60, 7, 7))
        new_weight[:, :3, :, :] = pretrained_weight

        # 创建新的state_dict，将预训练参数中的conv1.weight替换为修改后的权重
        net_state_dict = net.state_dict()
        net_state_dict['conv1.weight'] = new_weight

        # 更新state_dict的其他参数
        for key in state_dict:
            if key != 'conv1.weight':
                net_state_dict[key] = state_dict[key]

        # 加载更新后的模型参数
        net.load_state_dict(net_state_dict)


        # 替换最后一层
        net.fc = nn.Linear(net.fc.in_features, 1)

    else:
        predict_model = torch.load("/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_sl/resnet-50-kinetics.pth")
        state_dict = {k: v for k, v in predict_model.items() if k in net.state_dict().keys()}  # 寻找网络中公共层，并保留预训练参数
        net.fc = nn.Linear(net.fc.in_features, 1)
        # net_dict = net.state_dict()

        # net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
        # net.load_state_dict(net_dict)



    # 定义损失函数
    loss_function = nn.MSELoss()

    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=[0,1,2,3,4,5,6,7])



    net.to(device)



    Loss_list = []
    Loss_list_val = []
    stat=[]









    epochs =100
    best_mae =9999
    train_steps = len(train_loader)




    val_steps=len(validate_loader)

    data_list = []


    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0

        # -----------------添加下

        train_acc = 0.0  #

        val_loss=0.0

        # -----------------添加上




        train_bar = tqdm(train_loader)


        for count, (data, target,_) in enumerate(train_bar):

            target = target.to(device).float()
            target = target.view(-1, 1)

            if modeltype == "3dresnet50":
                logits = net(data)

            else:
                data_stacked = torch.stack(data, dim=1)  # 在第1维度堆叠
                data_stacked=data_stacked.view(data_stacked.size(0), -1, 224, 224)  

                logits = net(data_stacked)




            loss = loss_function(logits, target.to(device))
 
            running_loss += loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)



    



        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch

        predictions_list = []
        targets_list = []
        imgid_list=[]

        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images,val_labels,imgids = val_data


                val_labels = val_labels.to(device).float()
                val_labels = val_labels.view(-1, 1)

                

                #20张图片最初作为对象扔进来，方法：
                # outputs = net(val_images)

                if modeltype == "3dresnet50":
                    outputs = net(val_images)
                else:

                    #通道并起来
                    data_stacked = torch.stack(val_images, dim=1)  # 在第1维度堆叠
                    data_stacked=data_stacked.view(data_stacked.size(0), -1, 224, 224)  

                    outputs = net(data_stacked)





                predictions = outputs.cpu().numpy()
                targets = val_labels.cpu().numpy()

                # imgids=imgids.cpu().numpy()

                # 将预测结果和真实标签添加到列表中
                predictions_list.append(predictions)
                targets_list.append(targets)
                imgid_list.append(list(imgids))


                # -----------------添加下


                loss_val = loss_function(outputs, val_labels.to(device))
                val_loss+=loss_val.item()

                # -----------------添加上


                val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                           epochs,loss_val)

        data_list.append("epoch"+str(epoch)+'\n')
        for i in range(len(predictions_list)):
            row = f"Prediction: {predictions_list[i]}, Target: {targets_list[i]}, ImageID: {imgid_list[i]}"
            data_list.append(row)


        # 将所有预测结果和真实标签合并为一个数组
        predictions = np.concatenate(predictions_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)

        # 计算回归指标
        mse, mae, rmse, corr = compute_regression_metrics(predictions, targets)

        # 打印回归指标
        print("MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, Corr: {:.4f}".format(mse, mae, rmse, corr))

        stat.append("MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, Corr: {:.4f}".format(mse, mae, rmse, corr))


        data_list.append("MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, Corr: {:.4f}".format(mse, mae, rmse, corr)+'\n')


        Loss_list.append(running_loss / train_steps)

        Loss_list_val.append(val_loss / val_steps)







        print('[epoch %d] train_loss: %.3f  val_loss: %.3f' %
              (epoch + 1, running_loss / train_steps, val_loss / val_steps))





        if mae < best_mae:
            best_mae = mae
            torch.save(net.state_dict(), save_path_best)


        if epoch==epochs-1:
            torch.save(net.state_dict(), save_path_final)
            
            print(predictions)

            print(targets)


    print('Finished Training')


    with open('validation_results.txt', 'w') as file:
        for row in data_list:
            file.write(row + '\n')

    stat.append("best_mae: {:.4f}".format(best_mae))



    plt.subplot(1, 2, 2)
    x2 = range(0, epochs)
    y3 = Loss_list

    # y4=Loss_list_val

    plt.plot(x2, y3, '-', label="Train Loss")
    # plt.plot(x2, y4, '-', label="Test Loss")


    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.legend()

    plt.show()
    plt.savefig(path+'/'+"loss"+taskname+".jpg")



    writefile(path+'/'+"Loss_list"+taskname, Loss_list)

    writefile(path+'/'+"Loss_list_val"+taskname, Loss_list_val)




    writefile(path+'/'+"stat_"+taskname, stat)







if __name__ == '__main__':
    main()