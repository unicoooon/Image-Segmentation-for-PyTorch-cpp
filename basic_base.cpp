#include<torch/torch.h>
#include<opencv2/opencv.hpp>
#include <ogrsf_frmts.h>
#include <gdal_priv.h>
#include <ogr_geometry.h>
#include<iostream>
#include <string>
#include "utils.h"
using namespace std;
using namespace cv;
class linearNet:public torch::nn::Module
{
    public:
        int in_ch,out_ch;
        torch::nn::Conv2d conv1=nullptr;
        torch::nn::Conv2d conv2=nullptr;
        torch::nn::Conv2d fc=nullptr;
    public:
        linearNet(int inch,int outch):in_ch(inch),out_ch(outch),conv1(torch::nn::Conv2dOptions(in_ch,64,3).padding(1)),
                                      conv2(torch::nn::Conv2dOptions(64,128,3).padding(1)),
                                      fc(torch::nn::Conv2dOptions(128,out_ch,3).padding(1))
        {
            register_module("conv1",conv1);
            register_module("conv2",conv2);
            register_module("fc",fc);
        }
        torch::Tensor forward(torch::Tensor x)
        {
            x = conv1->forward(x);
            x = torch::relu(x);
            x = torch::relu(conv2->forward(x));
            x = fc->forward(x);
            return torch::log_softmax(x,1);
        }
};

float* trainData(const char *tifpath)
{
    float *res=new float[1000*1000*7];
    int count=0;
    cv::Mat data = GDAL2Mat(tifpath);
    for(int i=2000;i<3000;i++)
    {
        for(int j=2000;j<3000;j++)
        { 
            for(int k=0;k<7;k++)
            {
                res[count++]=data.at<Vec3b>(i,j)[k];
            }
        }
    }
    return res;
}
float* trainMask(const char *maskpath)
{
    cv::Mat data = cv::imread(maskpath,0);
    float *res = new float[1000*1000];
    int count=0;
    for(int i=2000;i<3000;i++)
    {
        for(int j=2000;j<3000;j++)
        {   
            res[count++]=data.at<uchar>(i,j);
        }
    }
    return res;
}
void trainModel(linearNet &net)
{
    torch::autograd::Variable train_x = torch::from_blob(trainData("/Users/yanlang/Desktop/121035/LC81210352014201LGN00_merge_align.tif"),{1,1000,1000,7},torch::TensorOptions().dtype(torch::kFloat32));
    train_x = torch::reshape(train_x,{1,7,1000,1000});
    torch::autograd::Variable train_y = torch::from_blob(trainMask("/Users/yanlang/Desktop/121035/shandong_14plant.png"),{1,1000,1000},torch::TensorOptions().dtype(torch::kFloat32));
    torch::optim::SGD optimizer(net.parameters(),/*lr=*/0.001);
    torch::Tensor pred,trainY;
    for(int epoch=0;epoch<30;epoch++)
    {
        pred = net.forward(train_x);
        auto loss = torch::nll_loss2d(pred,torch::_cast_Long(train_y));
        std::cout<<loss<<std::endl;
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}
void netpredict(linearNet nett)
{
    torch::autograd::Variable train_x = torch::from_blob(trainData("/Users/yanlang/Desktop/121035/LC81210352014201LGN00_merge_align.tif"),{1,1000,1000,7},torch::TensorOptions().dtype(torch::kFloat32));
    train_x = torch::reshape(train_x,{1,7,1000,1000});
    torch::Tensor out = nett.forward(train_x);
    torch::Tensor res = out.argmax(1).to(torch::kFloat32).reshape({1000,1000});
    std::cout<<get<0>(unique(res))<<std::endl; 
    float *rr = Tensor2Float(res);
    cv::Mat rg = cv::Mat(1000,1000,CV_8UC1);
    Float2Mat(rr,rg);
    cv::imwrite("/Users/yanlang/Desktop/bce.png",rg);
}

void tnll_loss(torch::Tensor pred,torch::Tensor label)
{
    auto tp = torch::nll_loss2d(pred,torch::_cast_Long(label));
    std::cout<<tp<<std::endl;
}

int main()
{   
    linearNet net(7,2);
    trainModel(net);
    netpredict(net);
    return 0;
}


