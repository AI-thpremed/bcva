import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

import torch
from PIL import Image
from torchvision import transforms


import torch.nn as nn

from os import path
from PIL import Image
import numpy as np
import pandas as pd
# from resnet_withnon import resnet50

from resnet import resnet50_multi,resnet50_multi_new,resnet50_multi_new_cbma,resnet50bam,resnet50se
import numpy as np

from torch.nn import DataParallel
import pandas as pd
from tqdm import tqdm
import random
from resnet import resnet50


from dataloader.image_transforms import Image_Transforms

from fusion_multi_model import Build_MultiModel_ShareBackbone,Build_MultiModel_ShareBackbone_3mod


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 计算回归指标的函数
def compute_regression_metrics(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    corr = r2_score(targets, predictions)
    return mse, mae, rmse, corr




data_res=[]






weights_path = "/vepfs/gaowh/backup/trsl/1023_2/1022_4_yd_big/resNet501022_4_yd_big_best.pth"




 
# IMAGE_PATH = '/vepfs/gaowh/tr_sl_vali/validation/'
IMAGE_PATH = '/vepfs/gaowh/tr_eyesl/'
 

test='val眼底散点图.txt'






label_path='/root/work2023/deep-learning-for-image-processing-master/data_set/TRSL_ALL/new'

label_path='/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_sl/final_label'

TRAIN_LISTS = ['train3.csv']
TEST_LISTS = ['test2.csv']
VAL_LISTS = ['validation.csv']



this_path = "/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_sl/"





# Load data

batch_size = 128

# 4 load dataset
train_transforms = Image_Transforms(mode='train', dataloader_id=1, square_size=256, crop_size=224).get_transforms()
val_transforms = Image_Transforms(mode='val', dataloader_id=1, square_size=256, crop_size=224).get_transforms()


NON_IMAGE_IN_NAMES = []
# Must be same length as NON_IMAGE_IN_NAMES
# 1: category to number
# 0: do nothing
# (a, b): (entry - a)/b
NON_IMAGE_TRANSFORMS = []




path_join = path.join







def write_csv(file_name: str, data: list,header:list) -> None:
    """
    将数据写入CSV文件。
    参数:
        file_name (str): CSV文件的路径及文件名。
        data (list): 包含字典的列表，每个字典代表一行数据。
    返回:
        None
    """
    try:
        df = pd.DataFrame(data)
        df.to_csv(file_name, index=False, header=True)
        print(f"数据已成功写入 {file_name}。")
    except Exception as e:
        print(f"写入数据时发生错误: {e}")










def get_random_images(image_files, max_img, seed=9):
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


 

def load_data(label_path, train_lists, img_path,
              batchsize, im_transforms,type):   


    train_sets = []
    train_loaders = []




    for train_list in train_lists:
        full_path_list = path_join(label_path,train_list)
        df = pd.read_csv(full_path_list)
        im_names = df['YD_PATH'].to_numpy()

        # im_labels = torch.tensor(df[classes].to_numpy(), dtype=torch.float)
        im_labels=df['etdrs'].to_numpy()
        im_path=df['PATH'].to_numpy()


    
        train_sets.append(CustomDataset(img_path, im_names, im_labels , im_path,im_transforms))
        train_loaders.append(torch.utils.data.DataLoader(train_sets[-1], batch_size=batchsize, shuffle=True,num_workers=8))
        print('Size for {0} = {1}'.format(train_list, len(im_names)))

        #这里的问题，因为原来的数据分了很多歌批次，不同的csv。这个就一个csv。原来是trainloader 数组拼出来。现在就取0就好了

    return train_loaders[0]





























def runpredict(LISTS):

    modeltype="3_allavg"



    validate_loader = load_data(label_path, TEST_LISTS, IMAGE_PATH, batch_size, val_transforms,'test')

  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   






    model = resnet50()

    # model = Build_MultiModel_ShareBackbone_3mod(backbone=modeltype)




    # # # change fc layer structure
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, 1)



    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])


    model.to(device)

    # 上面这个是最终结果。50跑了80轮

    # weights_path = "./resNet50_cataract_start_2.pth"   这个不行这个是个而分类的什么鬼。没用了
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

 



    data_res=[]


    data_res.append(['image','real','pred'])
    res=[]

    predictions_list = []
    targets_list = []
    # prediction
    model.eval()
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_data in validate_loader:

            val_images,val_labels,imgids = val_data


            val_labels = val_labels.to(device).float()
            val_labels = val_labels.view(-1, 1)

            

            #20张图片最初作为对象扔进来，方法：
            outputs = model(val_images)












            predictions = outputs.cpu().numpy()
            targets = val_labels.cpu().numpy()

            # # # 将预测结果和真实标签添加到列表中
            predictions_list.append(predictions)
            targets_list.append(targets)


            for idx, (cla) in enumerate(zip(predictions)):

                print("image: {} class:{} res:{}".format(imgids[idx],targets[idx][0],cla[0]   ))
      
                tempdata=[imgids[idx],float(val_labels[idx][0]),float(cla[0])]

                data_res.append(tempdata)



    # 将所有预测结果和真实标签合并为一个数组
    predictions = np.concatenate(predictions_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    mse, mae, rmse, corr = compute_regression_metrics(predictions, targets)

    write_csv(this_path+test, data_res,['image','real','pred'])

    print("MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, Corr: {:.4f}".format(mse, mae, rmse, corr))





def main():

#  TEST_LISTS = ['test.csv']
# VAL_LISTS = ['validation.csv']

    runpredict(VAL_LISTS)



if __name__ == '__main__':
    main()
