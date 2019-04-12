#include<torch/torch.h>
#include<opencv2/opencv.hpp>
#include <ogrsf_frmts.h>
#include <gdal_priv.h>
#include <ogr_geometry.h>
#include<iostream>
#include <string>
#include "vgg.h"
#include "segnet.h"
#include "pspnet.h"
#include "inception.h"
#include "densenet.h"
#include "fcn.h"
#include "deeplabv3.h"
using namespace std;
using namespace cv;
void trainNet(addNet &net)
{/*
  To train addNet with four multi-bands data;
  */
    cv::Mat data = GDAL2Mat("/Users/yanlang/Desktop/dyf/GF1_WFV3_E122.3_N45.5_20140807_L1A0000297900_rpc_reg_clip.tiff");
    cv::Mat mask = cv::imread("/Users/yanlang/Desktop/dyf/rice.png",0);
    std::vector<std::vector<cv::Mat>> res =DataLoader(data,mask,Size);
    torch::autograd::Variable train_data = torch::from_blob(GetFloatData(mat2floatData(res)),{Size,256,256,4});
    train_data = torch::transpose(torch::transpose(train_data,3,2),2,1);
    torch::autograd::Variable train_label =torch::from_blob(GetFloatMask(mat2floatMask(res)),{Size,256,256,1});
    std::cout<<std::get<0>(Tunique(train_label))<<std::endl;
    torch::Tensor prediction;
    torch::optim::Adam optimizer(net.parameters(),/*lr=*/0.001);
    for(int epoch=0;epoch<50;epoch++)
    {
        prediction = net.forward(train_data);
        std::cout<<prediction.sizes()<<std::endl;
        auto loss = torch::nll_loss2d(prediction,torch::_cast_Long(torch::squeeze(train_label)));
        std::cout<<loss<<std::endl;
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}
int main()
{
#if 0
    std::vector<int> vec={64,64,64,64};
    pspNet<torch::Tensor> net(4,2,vec);
    trainNet(net);
#endif
#if 0
    std::vector<int> vec={64,64,64,64,64};
    deeplabV3<torch::Tensor> net(4,2,vec);
    trainNet(net);
#endif
#if 0
    std::vector<int> vec={64,64,64,64,64};
    segNet<torch::Tensor> net(4,2,vec);
    trainNet(net);
#endif
#if 0
    std::vector<int> vec={64,64,64,64,64,64,64,64};
    inceptionNet<torch::Tensor> net(4,2,vec);
    trainNet(net);
#endif
#if 0
    std::vector<std::vector<int>> vec(4,std::vector<int>(1,0));
    vec[0]={64};
    vec[1]={128};
    vec[2]={256};
    vec[3]={512};
    denseNet<torch::Tensor> net(4,2,vec);
    trainNet(net);
#endif
    return 0;
}




