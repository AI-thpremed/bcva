import torch
import torch.nn as nn
from fusion.segating import SEGating
from fusion.average import project,SegmentConsensus
# from fusion.segating import SEGating
# from fusion.segating import SEGating
from fusion.nextvlad import NextVLAD


# from models.cv_models.swin_transformer import swin
# from models.cv_models.swin_transformer_v2 import swinv2
# # from models.cv_models.resnest import resnest50, resnest101
# from models.cv_models.convnext import convnext_tiny, convnext_small, convnext_base
from resnet import resnet50,resnet50_single_cbma

from models.FIT_Net import FITNet

from model3d import resnet



class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
        
# oct+oct
class Build_MultiModel_ShareBackbone(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, num_classes=1, use_gate=True, pretrained_modelpath='None'):
        super().__init__()
        self.input_dim = input_dim*20
        self.num_classes = num_classes
        self.use_gate = use_gate
        self.backbone=backbone



        self.gate = SegmentConsensus(self.input_dim,2048*5)


        self.lg1 = torch.nn.Linear(in_features=2048*5, out_features=2048)


        self.lg2 = torch.nn.Linear(in_features=2048*2, out_features=self.num_classes)

        self.lgavgavg = torch.nn.Linear(in_features=2048, out_features=self.num_classes)

        if backbone=='resnest50':

            self.model_oct = resnet50()
            self.model_oct.fc = Identity()
            self.model_yd = resnet50()
            self.model_yd.fc = Identity()
        elif backbone=='FITNet':
            self.model_oct = FITNet()
            self.model_oct.fc = torch.nn.Linear(self.model_oct.fc.in_features, 2048)

            # self.model_oct.fc = Identity()   #FITNet结果是1984 不是2048
            self.model_yd = resnet50()
            self.model_yd.fc = Identity()

        elif backbone=='resnest50cbma':
            self.model_oct = FITNet()
            self.model_oct.fc = torch.nn.Linear(self.model_oct.fc.in_features, 2048)

            # self.model_oct.fc = Identity()   #FITNet结果是1984 不是2048
            self.model_yd = resnet50_single_cbma()
            self.model_yd.fc = Identity()
        elif backbone=='resnest50cbma2':
            self.model_oct = resnet50_single_cbma()

            self.model_oct.fc = Identity()   #FITNet结果是1984 不是2048
            self.model_yd = resnet50_single_cbma()
            self.model_yd.fc = Identity()


        elif backbone=='resnest50next':
            self.model_oct = resnet50()

            self.model_oct.fc = Identity()   #FITNet结果是1984 不是2048
            self.model_yd = resnet50()
            self.model_yd.fc = Identity()


            self.next = NextVLAD(feature_size=input_dim,max_frames=2,cluster_size=32,output_dim=2048)
            self.lg3 = torch.nn.Linear(in_features=2048, out_features=self.num_classes)



        elif backbone=='resnest50stage':

            self.model_oct = resnet50()

            model_oct_path = "/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_sl/_resnet_sl_oct_multi_16fusion_average/resNet50_resnet_sl_oct_multi_16fusion_average_best.pth"
            self.model_oct.load_state_dict(torch.load(model_oct_path),strict=False)


            self.model_oct.fc = Identity()
            self.model_yd = resnet50()
            model_yd_path = "/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_sl/_resnet_sl_yd_all_20/resNet50_resnet_sl_yd_all_20_best.pth"
            self.model_yd.load_state_dict(torch.load(model_yd_path),strict=False)


            self.model_yd.fc = Identity()

        elif backbone=='resnest50_3d':

            self.model_oct = resnet.generate_model(model_depth=50,n_classes=1)
            net_dict = self.model_oct.state_dict()
            predict_model = torch.load("/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_sl/_resnet_273dresnet/resNet50_resnet_273dresnet_best.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in self.model_oct.state_dict().keys()}  # 寻找网络中公共层，并保留预训练参数
            net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
            self.model_oct.load_state_dict(net_dict)

            self.model_oct.fc = nn.Linear(self.model_oct.fc.in_features,1000)



            self.model_oct.fc = Identity()



            self.model_yd = resnet50()


            self.model_yd.fc = Identity()
        elif backbone=='hbsp_avgavg':

            self.model_hb = resnet50()
            self.model_hb.fc = Identity()
            self.model_sp = resnet50()
            self.model_sp.fc = Identity()
        elif backbone=='hbsp_avglinear':

            self.model_hb = resnet50()
            self.model_hb.fc = Identity()
            self.model_sp = resnet50()
            self.model_sp.fc = Identity()

        elif backbone=='hbsp_maxavg':

            self.model_hb = resnet50()
            self.model_hb.fc = Identity()
            self.model_sp = resnet50()
            self.model_sp.fc = Identity()

        print('init model:', backbone)

        if self.use_gate:
            print('use gate')
        else:
            print('no use gate')



    def forward(self, hb,sp):
        sp_output_list = []

 

        hb_output_list = []

        # 循环处理每个图像
        for i in range(len(hb)):
            image = hb[i]
            temp = self.model_hb(image)
            hb_output_list.append(temp)

        # 循环处理每个图像
        for i in range(len(sp)):
            image = sp[i]
            temp = self.model_sp(image)
            sp_output_list.append(temp)

        if self.backbone!='hbsp_maxavg':


            hb_avg_feats = torch.stack(hb_output_list, dim=1)
            hb_summed_feats = torch.sum(hb_avg_feats, dim=1)  # 逐个特征相加
            hb_avg_feats_mean = hb_summed_feats / len(hb_output_list)  # 计算平均特征

            sp_avg_feats = torch.stack(sp_output_list, dim=1)
            sp_summed_feats = torch.sum(sp_avg_feats, dim=1)  # 逐个特征相加
            sp_avg_feats_mean = sp_summed_feats / len(sp_output_list)  # 计算平均特征

            all=[]
            all.append(hb_avg_feats_mean)
            all.append(sp_avg_feats_mean)


            if self.backbone=='hbsp_avgavg':
                all_avg_feats = torch.stack(all, dim=1)
                all_summed_feats = torch.sum(all_avg_feats, dim=1)  # 逐个特征相加
                all_avg_feats_mean = all_summed_feats / len(all)  # 计算平均特征

                res = self.lgavgavg(all_avg_feats_mean)

            else:

                all_output = torch.cat(all, -1) # b, c1+c2
                res = self.lg2(all_output)
        else:

            sp_feats = torch.stack(sp_output_list, dim=1)
            sp_max_feats, _ = torch.max(sp_feats, dim=1)
            
            hb_feats = torch.stack(hb_output_list, dim=1)
            hb_max_feats, _ = torch.max(hb_feats, dim=1)
            all=[]
            all.append(hb_max_feats)
            all.append(sp_max_feats)
            all_avg_feats = torch.stack(all, dim=1)
            all_summed_feats = torch.sum(all_avg_feats, dim=1)  # 逐个特征相加
            all_avg_feats_mean = all_summed_feats / len(all)  # 计算平均特征

            res = self.lgavgavg(all_avg_feats_mean)




        return res

   



# img+img share backbone
class Build_MultiModel_ShareBackbone_octyd(nn.Module):
    def __init__(self, backbone='octyd_avglinear', input_dim=2048, num_classes=1, use_gate=True, pretrained_modelpath='None'):
        super().__init__()
        self.input_dim = input_dim*20
        self.num_classes = num_classes
        self.use_gate = use_gate
        self.backbone=backbone



        self.gate = SegmentConsensus(self.input_dim,2048*5)


        self.lg1 = torch.nn.Linear(in_features=2048*5, out_features=2048)


        self.lg2 = torch.nn.Linear(in_features=2048*2, out_features=self.num_classes)

        self.lgavgavg = torch.nn.Linear(in_features=2048, out_features=self.num_classes)


        self.model_oct = resnet50()
        self.model_oct.fc = Identity()
        self.model_yd = resnet50()
        self.model_yd.fc = Identity()



        print('init model:', backbone)

        if self.use_gate:
            print('use gate')
        else:
            print('no use gate')



    def forward(self, oct,yd):

 

        oct_output_list = []

        # 循环处理每个图像
        for i in range(len(oct)):
            image = oct[i]
            temp = self.model_oct(image)
            oct_output_list.append(temp)


        yd_output = self.model_yd(yd)


        if self.backbone=='octyd_maxavg':
            oct_feats = torch.stack(oct_output_list, dim=1)
            oct_max_feats, _ = torch.max(oct_feats, dim=1)
            all=[]
            all.append(oct_max_feats)
            all.append(yd_output)
            all_avg_feats = torch.stack(all, dim=1)
            all_summed_feats = torch.sum(all_avg_feats, dim=1)  # 逐个特征相加
            all_avg_feats_mean = all_summed_feats / len(all)  # 计算平均特征

            res = self.lgavgavg(all_avg_feats_mean)




        else:

        
            hb_avg_feats = torch.stack(oct_output_list, dim=1)
            hb_summed_feats = torch.sum(hb_avg_feats, dim=1)  # 逐个特征相加
            hb_avg_feats_mean = hb_summed_feats / len(oct_output_list)  # 计算平均特征


            all=[]
            all.append(hb_avg_feats_mean)
            all.append(yd_output)












            if self.backbone=='octyd_avgavg':
                all_avg_feats = torch.stack(all, dim=1)
                all_summed_feats = torch.sum(all_avg_feats, dim=1)  # 逐个特征相加
                all_avg_feats_mean = all_summed_feats / len(all)  # 计算平均特征

                res = self.lgavgavg(all_avg_feats_mean)

            else:

                all_output = torch.cat(all, -1) # b, c1+c2
                res = self.lg2(all_output)

        return res




       
# img+img share backbone
class Build_MultiModel_ShareBackbone_3mod_org(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, num_classes=1, use_gate=True, pretrained_modelpath='None'):
        super().__init__()
        self.input_dim = input_dim*20
        self.num_classes = num_classes
        self.use_gate = use_gate
        self.backbone=backbone




        self.lg2 = torch.nn.Linear(in_features=2048*2, out_features=self.num_classes)

        self.lgavgavg = torch.nn.Linear(in_features=2048, out_features=self.num_classes)

        self.model_hb = resnet50()
        self.model_hb.fc = Identity()
        self.model_sp = resnet50()
        self.model_sp.fc = Identity()
        self.model_yd = resnet50()
        self.model_yd.fc = Identity()



        print('init model:', backbone)

        if self.use_gate:
            print('use gate')
        else:
            print('no use gate')



    def forward(self, hb,sp,yd):
        sp_output_list = []

 
        yd_output_list=[]
        hb_output_list = []

        # 循环处理每个图像
        for i in range(len(hb)):
            image = hb[i]
            temp = self.model_hb(image)
            hb_output_list.append(temp)

        # 循环处理每个图像
        for i in range(len(sp)):
            image = sp[i]
            temp = self.model_sp(image)
            sp_output_list.append(temp)


        ydoutput = self.model_yd(yd)


        if self.backbone=='3_allmaxavg':


            

            #max
            hb_feats = torch.stack(hb_output_list, dim=1)
            hb_max_feats, _ = torch.max(hb_feats, dim=1)

            #max
            sp_feats = torch.stack(sp_output_list, dim=1)
            sp_max_feats, _ = torch.max(sp_feats, dim=1)

            
            all=[]
            all.append(hb_max_feats)
            all.append(sp_max_feats)
            all.append(ydoutput)
            all_avg_feats = torch.stack(all, dim=1)
            all_summed_feats = torch.sum(all_avg_feats, dim=1)  # 逐个特征相加
            all_avg_feats_mean = all_summed_feats / len(all)  # 计算平均特征
            res = self.lgavgavg(all_avg_feats_mean)

        else:


            hb_avg_feats = torch.stack(hb_output_list, dim=1)
            hb_summed_feats = torch.sum(hb_avg_feats, dim=1)  # 逐个特征相加
            hb_avg_feats_mean = hb_summed_feats / len(hb_output_list)  # 计算平均特征

            sp_avg_feats = torch.stack(sp_output_list, dim=1)
            sp_summed_feats = torch.sum(sp_avg_feats, dim=1)  # 逐个特征相加
            sp_avg_feats_mean = sp_summed_feats / len(sp_output_list)  # 计算平均特征

            # yd_avg_feats = torch.stack(yd_output_list, dim=1)
            # yd_summed_feats = torch.sum(yd_avg_feats, dim=1)  # 逐个特征相加
            # yd_avg_feats_mean = yd_summed_feats / len(yd_output_list)  # 计算平均特征

            all=[]
            all.append(hb_avg_feats_mean)
            all.append(sp_avg_feats_mean)
            # all.append(yd_avg_feats_mean)

            if self.backbone=='3_allavg':
                all.append(ydoutput)
                all_avg_feats = torch.stack(all, dim=1)
                all_summed_feats = torch.sum(all_avg_feats, dim=1)  # 逐个特征相加
                all_avg_feats_mean = all_summed_feats / len(all)  # 计算平均特征
                res = self.lgavgavg(all_avg_feats_mean)

            else:

                all_avg_feats = torch.stack(all, dim=1)
                all_summed_feats = torch.sum(all_avg_feats, dim=1)  # 逐个特征相加
                all_avg_feats_mean = all_summed_feats / len(all)  # 计算平均特征



                addyd=[]
                addyd.append(all_avg_feats_mean)
                # addyd.append(yd_avg_feats_mean)


                if self.backbone=='3_avglinear':
                    all_output = torch.cat(addyd, -1) # b, c1+c2
                    res = self.lg2(all_output)


                else:
                    addyd_avg = torch.stack(addyd, dim=1)
                    addyd_sum = torch.sum(addyd_avg, dim=1)  # 逐个特征相加
                    addyd_mean = addyd_sum / len(addyd)  # 计算平均特征

                    res = self.lgavgavg(addyd_mean)


        return res





# img+img share backbone
class Build_MultiModel_ShareBackbone_3mod(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, num_classes=1, use_gate=True, pretrained_modelpath='None'):
        super().__init__()
        self.input_dim = input_dim*20
        self.num_classes = num_classes
        self.use_gate = use_gate
        self.backbone=backbone




        self.lg2 = torch.nn.Linear(in_features=2048*2, out_features=self.num_classes)

        self.lgavgavg = torch.nn.Linear(in_features=2048, out_features=self.num_classes)

        self.model_hb = resnet50()
        self.model_hb.fc = Identity()
        self.model_sp = resnet50()
        self.model_sp.fc = Identity()
        self.model_yd = resnet50()
        self.model_yd.fc = Identity()



        print('init model:', backbone)

        if self.use_gate:
            print('use gate')
        else:
            print('no use gate')



    def forward(self, hb,sp,yd):
        sp_output_list = []

 
        yd_output_list=[]
        hb_output_list = []

        # 循环处理每个图像
        for i in range(len(hb)):
            image = hb[i]
            temp = self.model_hb(image)
            hb_output_list.append(temp)

        # 循环处理每个图像
        for i in range(len(sp)):
            image = sp[i]
            temp = self.model_sp(image)
            sp_output_list.append(temp)


        # 循环处理每个图像
        for i in range(len(yd)):
            image = yd[i]
            temp = self.model_yd(image)
            yd_output_list.append(temp)

        # ydoutput = self.model_yd(yd)


        if self.backbone=='3_allmaxavg':


            

            #max
            hb_feats = torch.stack(hb_output_list, dim=1)
            hb_max_feats, _ = torch.max(hb_feats, dim=1)

            #max
            sp_feats = torch.stack(sp_output_list, dim=1)
            sp_max_feats, _ = torch.max(sp_feats, dim=1)

            
            all=[]
            all.append(hb_max_feats)
            all.append(sp_max_feats)
            # all.append(ydoutput)
            all_avg_feats = torch.stack(all, dim=1)
            all_summed_feats = torch.sum(all_avg_feats, dim=1)  # 逐个特征相加
            all_avg_feats_mean = all_summed_feats / len(all)  # 计算平均特征
            res = self.lgavgavg(all_avg_feats_mean)

        else:


            hb_avg_feats = torch.stack(hb_output_list, dim=1)
            hb_summed_feats = torch.sum(hb_avg_feats, dim=1)  # 逐个特征相加
            hb_avg_feats_mean = hb_summed_feats / len(hb_output_list)  # 计算平均特征

            sp_avg_feats = torch.stack(sp_output_list, dim=1)
            sp_summed_feats = torch.sum(sp_avg_feats, dim=1)  # 逐个特征相加
            sp_avg_feats_mean = sp_summed_feats / len(sp_output_list)  # 计算平均特征

            yd_avg_feats = torch.stack(yd_output_list, dim=1)
            yd_summed_feats = torch.sum(yd_avg_feats, dim=1)  # 逐个特征相加
            yd_avg_feats_mean = yd_summed_feats / len(yd_output_list)  # 计算平均特征

            all=[]
            all.append(hb_avg_feats_mean)
            all.append(sp_avg_feats_mean)
            all.append(yd_avg_feats_mean)

            if self.backbone=='3_allavg':
                # all.append(ydoutput)
                all_avg_feats = torch.stack(all, dim=1)
                all_summed_feats = torch.sum(all_avg_feats, dim=1)  # 逐个特征相加
                all_avg_feats_mean = all_summed_feats / len(all)  # 计算平均特征
                res = self.lgavgavg(all_avg_feats_mean)

            else:

                all_avg_feats = torch.stack(all, dim=1)
                all_summed_feats = torch.sum(all_avg_feats, dim=1)  # 逐个特征相加
                all_avg_feats_mean = all_summed_feats / len(all)  # 计算平均特征



                addyd=[]
                addyd.append(all_avg_feats_mean)
                addyd.append(yd_avg_feats_mean)


                if self.backbone=='3_avglinear':
                    all_output = torch.cat(addyd, -1) # b, c1+c2
                    res = self.lg2(all_output)


                else:
                    addyd_avg = torch.stack(addyd, dim=1)
                    addyd_sum = torch.sum(addyd_avg, dim=1)  # 逐个特征相加
                    addyd_mean = addyd_sum / len(addyd)  # 计算平均特征

                    res = self.lgavgavg(addyd_mean)


        return res




# yd+yd
class Build_MultiModel_ShareBackbone_yd(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, num_classes=1, use_gate=True, pretrained_modelpath='None'):
        super().__init__()
        self.input_dim = input_dim*20
        self.num_classes = num_classes
        self.use_gate = use_gate
        self.backbone=backbone



        self.gate = SegmentConsensus(self.input_dim,2048*5)


        self.lg1 = torch.nn.Linear(in_features=2048*5, out_features=2048)


        self.lg2 = torch.nn.Linear(in_features=2048*2, out_features=self.num_classes)

        self.lgavgavg = torch.nn.Linear(in_features=2048, out_features=self.num_classes)

        self.model = resnet50()
        self.model.fc = Identity()



        print('init model:', backbone)

        if self.use_gate:
            print('use gate')
        else:
            print('no use gate')



    def forward(self, yd):
        yd_output_list = []

 


        # 循环处理每个图像
        for i in range(len(yd)):
            image = yd[i]
            temp = self.model(image)
            yd_output_list.append(temp)

        if self.backbone=='avg':


            yd_avg_feats = torch.stack(yd_output_list, dim=1)
            yd_summed_feats = torch.sum(yd_avg_feats, dim=1)  # 逐个特征相加
            yd_avg_feats_mean = yd_summed_feats / len(yd_output_list)  # 计算平均特征

            res = self.lgavgavg(yd_avg_feats_mean)


        elif self.backbone=='max':
            yd_feats = torch.stack(yd_output_list, dim=1)
            yd_max_feats, _ = torch.max(yd_feats, dim=1)

            res = self.lgavgavg(yd_max_feats)


        else:


            all_output = torch.cat(yd_output_list, -1) # b, c1+c2
            res = self.lg2(all_output)





        return res




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


