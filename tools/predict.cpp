// Copyright 2014 BVLC and contributors.
//
// This is a simple script that allows one to quickly test a network whose
// structure is specified by text format protocol buffers, and whose parameter
// are loaded from a pre-trained network.
// Usage:
//    test_net net_proto pretrained_net_proto iterations [CPU/GPU]

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <queue>
#include <map>

#include "caffe/caffe.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;

#define DEBUG 0

Net<float> caffe_test_net("predict.prototxt");

int predict(queue<Mat> frames,string net_proto,string pretrained_model,string mean_file,string mode,int device_id) ;

int main(int argc, char const *argv[])
{
  string net_proto = argv[1];
  string pretrained_model = argv[2];
  string mean_file = argv[3];
  string mode;
  int device_id;
  if (argc < 4 || argc > 6) {
    LOG(ERROR) << "test_net net_proto pretrained_net_proto iterations "
        << "[CPU/GPU] [Device ID]";
    return 1;
  }
  if (argc >= 5 && strcmp(argv[4], "GPU") == 0) {
    mode = argv[4]; 
    device_id = 0;
    if (argc == 6) {
      device_id = atoi(argv[5]);
    }
  } else {
    string mode = "CPU";
  }

  // config the predictor
  Caffe::set_phase(Caffe::TEST);
  if (strcmp(mode.c_str(), "GPU") == 0) {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(device_id);
    LOG(ERROR) << "Using GPU #" << device_id;
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }
  // 建立网络，复制网络参数
  caffe_test_net.CopyTrainedLayersFrom(pretrained_model.c_str());

  //读取均值文件
  Blob<float> data_mean_;
  LOG(INFO) << "Loading mean_value file from " << mean_file;
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
  data_mean_.FromProto(blob_proto);
  CHECK_EQ(data_mean_.num(), 1);
  CHECK_EQ(data_mean_.channels(), 3);
  CHECK_EQ(data_mean_.length(), 16);
  CHECK_EQ(data_mean_.height(), 128);
  CHECK_EQ(data_mean_.width(), 171);
  const float * mean_value = data_mean_.cpu_data();
  int mean_size = data_mean_.num()*data_mean_.channels()*data_mean_.length()*data_mean_.height()*data_mean_.width();

  // read files and predict
  // 采用滑窗法
  int length = 16;// 图片序列长度固定为16
  queue<Mat> frames;// 用于储存图片序列的队列
  ifstream input_list("predict.txt");// 输入文件列表
  CHECK(input_list.is_open())<<"can not open file predict.txt";
  char filename[50];
  char dirName[50];
  char outFileName[50];
  int start_frame;
  int ignore;
  int frm_count = 0;

  // label与动作名之间的对应关系
  map<int,string> map_label;
  map_label[0] = "place";
  map_label[1] = "push";
  map_label[2] = "screw";
  map_label[3] = "turndown";
  map_label[4] = "press";

  while(input_list>>dirName>>start_frame>>ignore)
  {
    for(int i = start_frame; i < start_frame+length; i++)
    {
        sprintf(filename,"%s/%06d.jpg",dirName,i);
        sprintf(outFileName,"output/predict_frm/%06d.jpg",++frm_count);
        Mat tempImg = imread(filename);
        if(!tempImg.data) LOG(ERROR) << "Could not open or find file" << filename;
        if(frames.size()<length)
        {
          frames.push(tempImg);
          imwrite(outFileName,tempImg);
        } 
        else if(frames.size()>=length) 
        {
          frames.pop();
          frames.push(tempImg);
          int label = predict(frames,net_proto,pretrained_model,mean_file,mode,device_id);
          if (label>=0&&label<=4) {
            string action_name = map_label[label];
            putText(tempImg,action_name,Point(50,50),FONT_HERSHEY_PLAIN,2,Scalar(255,0,255));
          }
          imwrite(outFileName,tempImg);
        }
    }
    
  }

  return 0;
}


int predict(queue<Mat> frames,string net_proto,string pretrained_model,string mean_file,string mode,int device_id) 
{   
    int num = 1;
    int channels = 3;
    int length = 16;
    int height = 128;
    int width = 171;
    int crop_size = 112;
    unsigned long size = num*channels*length*crop_size*crop_size;
    shared_ptr<Blob<float> > top_data;//整理原始输入后得到的输入(经过剪切) num channels length crop_size crop_size
    //Blob<float>* top_data;//整理原始输入后得到的输入(经过剪切) num channels length crop_size crop_size
    top_data.reset(new Blob<float>(
        num, channels, length,
        crop_size, crop_size));
    #ifdef DEBUG
      LOG(ERROR)<<"changed";
      LOG(ERROR)<<"top_data.shape is:"<<top_data->num()<<'x'<<top_data->channels()<<'x'<<top_data->length()<<'x'<<top_data->height()<<'x'<<top_data->width();
    #endif
    int h_off = (height - crop_size) / 2;//剪切时的偏移量，在测试时直接取中心
    int w_off = (width - crop_size) / 2;
  
    for(int i = 0; i < length; i++)
    {
        Mat tempImg = frames.front();
        frames.pop();
        CHECK(tempImg.data)<<"can not read image from queue";
        #ifdef DEBUG
          LOG(ERROR)<<"one value of image:"<<tempImg.at<Vec3f>(80,60)[0];
        #endif
        for(int c = 0;c < channels;++c)
          for(int h = 0;h < crop_size;++h)
            for(int w = 0;w < crop_size;++w)
            {
              int top_index = ((c * length + i) * crop_size + h)
                              * crop_size + w;
              int data_index = ((c * length + i) * height + h + h_off) * width + w + w_off;
              CHECK_GT(size,top_index)<<"top_data segmentation fault";
              //CHECK_GT(mean_size,data_index)<<"mean_data_ segmentation fault";
              int datum_element = tempImg.at<Vec3b>(h+h_off,w+w_off)[c];
              top_data->mutable_cpu_data()[top_index] = datum_element - mean_value[data_index]; 
            }
    }
    #ifdef DEBUG
      LOG(ERROR)<<"one value of top data mutable:"<<top_data->mutable_cpu_data()[10000];
      LOG(ERROR)<<"one value of top data cpu:"<<top_data->cpu_data()[10000];
      LOG(ERROR)<<top_data->num()<<'x'<<top_data->channels()<<'x'<<top_data->length()<<'x'<<top_data->height()<<'x'<<top_data->width();
    #endif

    shared_ptr<Blob<float> > data_blob = caffe_test_net.blob_by_name("data");
    data_blob->set_cpu_data(top_data->mutable_cpu_data());

    const vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled();
    const shared_ptr<Blob<float> > feature_blob = caffe_test_net.blob_by_name("prob");
    LOG(ERROR) << feature_blob.get()->num() << 'x'<<feature_blob.get()->channels() << 'x'<<feature_blob.get()->length()<<
              'x'<<feature_blob.get()->width() << 'x'<<feature_blob.get()->height();
        
    float max = -1;
    int maxIndex = -1;
    for(int j = 0; j < 5; j++)
    {
        float data = feature_blob.get()->cpu_data()[j];
        #ifdef DEBUG
          LOG(ERROR) << data;
        #endif
        if(data>max)
        {
          max = data;
          maxIndex = j;
        }
    }
    if(max<0.9) maxIndex = -1;
    LOG(ERROR) << maxIndex;      
    
  return maxIndex;
}
