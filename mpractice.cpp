#include<torch/torch.h>
#include<opencv2/opencv.hpp>
#include <ogrsf_frmts.h>
#include <gdal_priv.h>
#include <ogr_geometry.h>
#include<iostream>
#include <string>
#include "utils.h"
#include "resnet50.h"
using namespace std;
using namespace cv;
int main()
{
    std::unordered_map<std::string,std::vector<int>> mp;
    mp["one"]={64,64};
    mp["two"]={64,64};
    mp["three"]={64,64};
    mp["conn"]={64,64};
    resnet50<torch::Tensor> res(64,64,mp);
    torch::Tensor out =res.forward(torch::randn({1,64,256,256}));
    std::cout<<out.sizes()<<std::endl;
    return 0;
}
