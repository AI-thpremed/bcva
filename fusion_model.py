import torch
import torch.nn as nn
from fusion.segating import SEGating
from fusion.average import project,SegmentConsensus
# from fusion.segating import SEGating
# from fusion.segating import SEGating


# from models.cv_models.swin_transformer import swin
# from models.cv_models.swin_transformer_v2 import swinv2
# # from models.cv_models.resnest import resnest50, resnest101
# from models.cv_models.convnext import convnext_tiny, convnext_small, convnext_base
from resnet import resnet50,resnet50_single_cbma

# pretrained_models_dir = {
#         'resnet50': '/data/chencc/pretrained_models/image_models/resnet50-19c8e357.pth',
#         'resnest50': '/data/chencc/pretrained_models/image_models/resnest50-528c19ca.pth',

#         'convnexttiny': '/data/chencc/pretrained_models/image_models/convnext_tiny_22k_1k_224.pth',
#         'convnextsmall': '/data/chencc/pretrained_models/image_models/convnext_small_22k_1k_224.pth',
#         'convnextbase': '/data/chencc/pretrained_models/image_models/convnext_base_22k_1k_224.pth',
#         'convnexttiny384': '/data/chencc/pretrained_models/image_models/convnext_tiny_22k_1k_384.pth',
#         'convnextsmall384': '/data/chencc/pretrained_models/image_models/convnext_small_22k_1k_384.pth',
#         'convnextbase384': '/data/chencc/pretrained_models/image_models/convnext_base_22k_1k_384.pth',

#         'swintiny': '/data/chencc/pretrained_models/image_models/swin_tiny_patch4_window7_224_22kto1k_finetune.pth',
#         'swinsmall': '/data/chencc/pretrained_models/image_models/swin_small_patch4_window7_224_22kto1k_finetune.pth',
#         'swinbase': '/data/chencc/pretrained_models/image_models/swin_base_patch4_window7_224_22kto1k.pth',
#         'swinbase384': '/data/chencc/pretrained_models/image_models/swin_base_patch4_window12_384_22kto1k.pth',

#         'swinv2tiny': '/data/chencc/pretrained_models/image_models/swinv2_tiny_patch4_window16_256.pth',
#         'swinv2small': '/data/chencc/pretrained_models/image_models/swinv2_small_patch4_window16_256.pth',
#         'swinv2base': '/data/chencc/pretrained_models/image_models/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth',
#         'swinv2base384': '/data/chencc/pretrained_models/image_models/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth',
#     }



class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
        
# img+img share backbone
class Build_MultiModel_ShareBackbone(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, num_classes=1, use_gate=True, pretrained_modelpath='None'):
        super().__init__()
        self.input_dim = input_dim*12
        self.num_classes = num_classes
        self.use_gate = use_gate


        # self.gate = SEGating(self.input_dim)
        # self.cate2 = nn.Sequential(nn.Linear(in_features=self.input_dim, out_features=512), nn.BatchNorm1d(512), nn.ReLU())
        # self.lg2 = torch.nn.Linear(in_features=512, out_features=self.num_classes)


        # self.gate = SegmentConsensus(self.input_dim,2048*3)


        # self.lg1 = torch.nn.Linear(in_features=2048*3, out_features=2048)
        self.gate = SegmentConsensus(self.input_dim,2048*6)


        self.lg1 = torch.nn.Linear(in_features=2048*6, out_features=2048*3)
        # self.lg2 = torch.nn.Linear(in_features=2048*8, out_features=2048*7)
        # self.lg3 = torch.nn.Linear(in_features=2048*7, out_features=2048*3)
        self.lg4 = torch.nn.Linear(in_features=2048*3, out_features=2048)



        self.lg5 = torch.nn.Linear(in_features=2048, out_features=self.num_classes)

        if backbone=="resnet50_single_cbma":

            self.model = resnet50_single_cbma()
        else:
            self.model = resnet50()

        self.model.fc = Identity()
        print('init model:', backbone)

        if self.use_gate:
            print('use gate')
        else:
            print('no use gate')


    def forward(self, x):




        encoder_output_list = []

        # 循环处理每个图像
        for i in range(len(x)):
            image = x[i]


            temp = self.model(image)
            encoder_output_list.append(temp)


        # res = self.fc(avg_feats_mean)
        output = torch.cat(encoder_output_list, -1) # b, c1+c2
        output = self.gate(output)

        output = self.lg1(output)

        # output = self.cate2(output)
        # output = self.lg2(output)
        # output = self.lg3(output)
        output = self.lg4(output)
        output = self.lg5(output)
        # output = self.lg3(output)


        return output

from train_multi_avg import load_data_multi
from dataloader.image_transforms import Image_Transforms
from tqdm import tqdm

if __name__ == '__main__':
    model = Build_MultiModel_ShareBackbone()





    label_path='/root/work2023/deep-learning-for-image-processing-master/data_set/TRSL_ALL'




    IMAGE_PATH = '/vepfs/gaowh/tr_eyesl/'

    TRAIN_LISTS = ['train.csv']
    TEST_LISTS = ['test.csv']
    val_transforms = Image_Transforms(mode='val', dataloader_id=1, square_size=256, crop_size=224).get_transforms()






    validate_loader = load_data_multi(label_path, TEST_LISTS, IMAGE_PATH, 16, val_transforms,'test')

    val_bar = tqdm(validate_loader)
    for val_data in val_bar:
        val_images,val_labels,imgids = val_data


        outputs = model(val_images)
        print(outputs.shape)



    # input = torch.randn(4, 40960)  # 替换为您的实际输入数据
    # segment_consensus = SegmentConsensus(40960, 256)
    # output = segment_consensus(input)


    # inputs = torch.randn(2, 3, 224, 224)
    # output = model(inputs, inputs)
    # print(output)

