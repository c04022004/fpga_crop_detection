/*
-- (c) Copyright 2018 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
*/

#include <algorithm>
#include <vector>
#include <iterator>
#include <atomic>
#include <queue>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <mutex>
#include <zconf.h>
#include <thread>
#include <sys/stat.h>
#include <dirent.h>
#include <sys/time.h>

#include <dnndk/dnndk.h>

#include "utils.h"


using namespace std;
using namespace cv;
using namespace std::chrono;


#define INPUT_NODE "layer0_conv"
#define NUM_YOLO_THREAD 1

int idxInputImage = 0;  // frame index of input video
int idxShowImage = 0;   // next frame index to be displayed
bool bReading = true;   // flag of reding input frame

// Performance Metrics
chrono::system_clock::time_point start_time,end_time;
// typedef pair<int, chrono::system_clock::time_point> timePair;
// vector<timePair> in_pair;
// vector<timePair> out_pair;
// bool tpaircomp (const timePair &n1, const timePair &n2) {
//     return (n1.first > n2.first);
// }

typedef pair<int, Mat> imagePair;
class paircomp {
    public:
    bool operator()(const imagePair &n1, const imagePair &n2) const {
        if (n1.first == n2.first) {
            return (n1.first > n2.first);
        }

        return n1.first > n2.first;
    }
};

// mutex for protection of input frames queue
mutex mtxQueueInput;
// mutex for protection of display frmaes queue
mutex mtxQueueShow;
// input frames queue
queue<pair<int, Mat>> queueInput;
// display frames queue
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow;

/**
 * @brief Feed input frame into DPU for process
 *
 * @param task - pointer to DPU Task for YOLO-v3 network
 * @param frame - pointer to input frame
 * @param mean - mean value for YOLO-v3 network
 *
 * @return none
 */
void setInputImageForYOLO(DPUTask* task, const Mat& frame, float* mean) {
    Mat img_copy;
    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);
    int size = dpuGetInputTensorSize(task, INPUT_NODE);
    int8_t* data = dpuGetInputTensorAddress(task, INPUT_NODE);

    image img_new = load_image_cv(frame);
    image img_yolo = letterbox_image(img_new, width, height);

    vector<float> bb(size);
    for(int b = 0; b < height; ++b) {
        for(int c = 0; c < width; ++c) {
            for(int a = 0; a < 3; ++a) {
                bb[b*width*3 + c*3 + a] = img_yolo.data[a*height*width + b*width + c];
            }
        }
    }

    float scale = dpuGetInputTensorScale(task, INPUT_NODE);

    for(int i = 0; i < size; ++i) {
        data[i] = int(bb.data()[i]*scale);
        if(data[i] < 0) data[i] = 127;
    }

    free_image(img_new);
    free_image(img_yolo);
}

/**
 * @brief Thread entry for reading image frame from the input video file
 *
 * @param fileName - pointer to video file name
 *
 * @return none
 */
void readFrame(const char *fileName) {

    static int loop = 3;

    ifstream inputFile(fileName);
    if(!inputFile) {
        cout << "File list could not be opened\n";
        exit(-1);
    }
    vector<string> fileList;
    string line;
    while(getline(inputFile, line)) {
        fileList.push_back(line);
    }
    cout << "Number of files to be analyzed: " << fileList.size() << "\n";

    vector<string>::const_iterator it(fileList.begin());
    vector<string>::const_iterator end(fileList.end());

    // Playing around with the lock to fill the buffer first
    bool init = false;
    mtxQueueInput.lock();
    start_time = chrono::system_clock::now();
    
    while (loop>0) {
        loop--;

        while (true) {
            Mat img;
            if(queueInput.size()<600 && it!=end){
                img = imread(it->c_str());
                if(!img.data){break;}else{++it;}
                if(init){mtxQueueInput.lock();}
                queueInput.push(make_pair(idxInputImage++, img));
                if(init){mtxQueueInput.unlock();}
                cout << "Queue size: " << queueInput.size() << endl;
            }else if(!init){
                mtxQueueInput.unlock();
                init = true;
                start_time = chrono::system_clock::now();
            }else if(it==end){
                bReading = false;
                return;
            }else{
                usleep(10);
            }
        }
    }
    exit(0);
}

/**
 * @brief Thread entry for displaying image frames
 *
 * @param  none
 * @return none
 */
void displayFrame() {
    Mat frame;

    while (true) {
        mtxQueueShow.lock();

        if (queueShow.empty()) {
            mtxQueueShow.unlock();
            if(bReading){continue;}else{break;}
            usleep(10);
        } else if (idxShowImage == queueShow.top().first) {
            auto show_time = chrono::system_clock::now();
            stringstream buffer;
            frame = queueShow.top().second;

            auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
            buffer << fixed << setprecision(1)
                   << (float)queueShow.top().first / (dura / 1000000.f);
            string a = buffer.str() + " FPS";
            cout << a << endl;
            cv::putText(frame, a, cv::Point(10, 15), 1, 1, cv::Scalar{240, 240, 240},1);
            // cv::imshow("Yolo@Xilinx DPU", frame);
            // cv::imwrite(fname.c_str(), img);

            idxShowImage++;
            queueShow.pop();
            mtxQueueShow.unlock();
            // if (waitKey(1) == 'q') {
            //     bReading = false;
            //     exit(0);
            // }
        } else {
            mtxQueueShow.unlock();
        }
    }
}


/**
 * @brief Post process after the runing of DPU for YOLO-v3 network
 *
 * @param task - pointer to DPU task for running YOLO-v3
 * @param frame
 * @param sWidth
 * @param sHeight
 *
 * @return none
 */
void postProcess(DPUTask* task, Mat& frame, int sWidth, int sHeight){

    /*output nodes of YOLO-v3 */
    const vector<string> outputs_node = {"layer81_conv", "layer93_conv", "layer105_conv"};
    // const vector<string> outputs_node = {"layer15_conv", "layer22_conv"};
    // const vector<string> outputs_node = {"layer15_conv", "layer22_conv", "layer29_conv"};
    // const vector<string> outputs_node = {"layer14_conv", "layer21_conv", "layer28_conv"};

    vector<vector<float>> boxes;
    for(size_t i = 0; i < outputs_node.size(); i++){
        string output_node = outputs_node[i];
        int channel = dpuGetOutputTensorChannel(task, output_node.c_str());
        int width = dpuGetOutputTensorWidth(task, output_node.c_str());
        int height = dpuGetOutputTensorHeight(task, output_node.c_str());

        int sizeOut = dpuGetOutputTensorSize(task, output_node.c_str());
        int8_t* dpuOut = dpuGetOutputTensorAddress(task, output_node.c_str());
        float scale = dpuGetOutputTensorScale(task, output_node.c_str());
        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);

        /* Store every output node results */
        get_output(dpuOut, sizeOut, scale, channel, height, width, result);

        /* Store the object detection frames as coordinate information  */
        detect(boxes, result, channel, height, width, i, sHeight, sWidth);
    }

    /* Restore the correct coordinate frame of the original image */
    correct_region_boxes(boxes, boxes.size(), frame.cols, frame.rows, sWidth, sHeight);

    /* Apply the computation for NMS */
    // cout << "boxes size: " << boxes.size() << endl;
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    float h = frame.rows;
    float w = frame.cols;
    for(size_t i = 0; i < res.size(); ++i) {
        float xmin = (res[i][0] - res[i][2]/2.0) * w + 1.0;
        float ymin = (res[i][1] - res[i][3]/2.0) * h + 1.0;
        float xmax = (res[i][0] + res[i][2]/2.0) * w + 1.0;
        float ymax = (res[i][1] + res[i][3]/2.0) * h + 1.0;

        if(res[i][res[i][4] + 6] > CONF ) {

            // Sanity Check
            if ( (xmin<0&&xmax<0) || (xmin>w&&xmax>w) || (ymin<0&&ymax<0) || (ymin>h&&ymax>h)){
                continue;
            }

            // Results output
            int type = res[i][4];  // result of classification
            string type_names[7] = {"Broccoli","Cauliflower","Spectralon_15%", "Spectralon_30%",
                    "Spectralon_60%", "ColorChecker_C", "ColorChecker_Gy"}; // as in lady.names
            // cout << fixed << setprecision(5) << res[i][type + 6];
            // cout<<"\t"<<type_names[type]<<"\t";
            // cout<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<endl;

            // Mark the detction results on the image with OpenCV
            // Scalar(a,b,c)  We would be defining a BGR color such as: Blue=a, Green=b and Red=c
            if (type==0) {          // Broccoli
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0, 255, 0), 1, 1, 0);
            } else if (type==1) {    // Cauliflower
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(255, 255, 255), 1, 1, 0);
            } else {                 // Others
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0 ,255, 255), 1, 1, 0);
            }
        }
    }
}

vector<vector<float>> postProcessResults(DPUTask* task, Mat& frame, int sWidth, int sHeight){

    /*output nodes of YOLO-v3 */
    const vector<string> outputs_node = {"layer81_conv", "layer93_conv", "layer105_conv"};
    // const vector<string> outputs_node = {"layer15_conv", "layer22_conv"};
    // const vector<string> outputs_node = {"layer15_conv", "layer22_conv", "layer29_conv"};
    // const vector<string> outputs_node = {"layer14_conv", "layer21_conv", "layer28_conv"};

    vector<vector<float>> boxes;
    for(size_t i = 0; i < outputs_node.size(); i++){
        string output_node = outputs_node[i];
        int channel = dpuGetOutputTensorChannel(task, output_node.c_str());
        int width = dpuGetOutputTensorWidth(task, output_node.c_str());
        int height = dpuGetOutputTensorHeight(task, output_node.c_str());

        int sizeOut = dpuGetOutputTensorSize(task, output_node.c_str());
        int8_t* dpuOut = dpuGetOutputTensorAddress(task, output_node.c_str());
        float scale = dpuGetOutputTensorScale(task, output_node.c_str());
        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);

        /* Store every output node results */
        get_output(dpuOut, sizeOut, scale, channel, height, width, result);

        /* Store the object detection frames as coordinate information  */
        detect(boxes, result, channel, height, width, i, sHeight, sWidth);
    }

    /* Restore the correct coordinate frame of the original image */
    correct_region_boxes(boxes, boxes.size(), frame.cols, frame.rows, sWidth, sHeight);

    /* Apply the computation for NMS */
    // cout << "boxes size: " << boxes.size() << endl;
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    return res;
}


/**
 * @brief Thread entry for running YOLO-v3 network on DPU for acceleration
 *
 * @param task - pointer to DPU task for running YOLO-v3
 * @param img
 *
 * @return none
 */
void runYOLO(DPUTask* task, Mat& img) {
    /* mean values for YOLO-v3 */
    float mean[3] = {0.0f, 0.0f, 0.0f};

    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);


    /* feed input frame into DPU Task with mean value */
    start_time = chrono::system_clock::now();
    setInputImageForYOLO(task, img, mean);
    end_time = chrono::system_clock::now();
    cout << "setInputImageForYOLO(): " << ((duration<double>)(end_time-start_time)).count() << "s" << endl;

    /* invoke the running of DPU for YOLO-v3 with c++ timers*/
    // tesing loop
    // for(int i=0;i<1000;i++){
    //     struct timeval stop, start;
    //     gettimeofday(&start, NULL);
    //     dpuRunTask(task);
    //     gettimeofday(&stop, NULL);
    //     printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec); 
    // }
    auto start_time = chrono::system_clock::now();
    dpuRunTask(task);
    auto end_dpu_time = chrono::system_clock::now();
    postProcess(task, img, width, height);
    auto end_cpu_time = chrono::system_clock::now();
    duration<double> dpu_seconds = end_dpu_time - start_time;
    duration<double> cpu_seconds = end_cpu_time - end_dpu_time;
    cout << "DPU time: " << dpu_seconds.count() << "s" << endl;
    cout << "CPU time: " << cpu_seconds.count() << "s" << endl;
}


/**
 * @brief Thread entry for running YOLO-v3 network on DPU for acceleration
 *
 * @param task - pointer to DPU task for running YOLO-v3
 *
 * @return none
 */
void runYOLO_video(DPUTask* task) {
    /* mean values for YOLO-v3 */
    float mean[3] = {0.0f, 0.0f, 0.0f};

    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);

    while (true) {
        pair<int, Mat> pairIndexImage;

        mtxQueueInput.lock();
        if (queueInput.empty()) {
            mtxQueueInput.unlock();
            if(bReading){continue;}else{break;}
        } else {
            /* get an input frame from input frames queue */
            pairIndexImage = queueInput.front();
            queueInput.pop();
            mtxQueueInput.unlock();
        }
        // cout << "Queue size: " << queueInput.size() << endl;
        // in_pair.push_back(make_pair(pairIndexImage.first, chrono::system_clock::now()));
        chrono::system_clock::time_point start_pipe, end_pile;

        vector<vector<float>> res;
        /* feed input frame into DPU Task with mean value */
        start_pipe = chrono::system_clock::now();
        setInputImageForYOLO(task, pairIndexImage.second, mean);

        /* invoke the running of DPU for YOLO-v3 */
        dpuRunTask(task);

        // testing loop for max load
        // for(int i=0; i<100; i++){
        //     dpuRunTask(task);
        // }

        postProcess(task, pairIndexImage.second, width, height);
        // out_pair.push_back(make_pair(pairIndexImage.first, chrono::system_clock::now()));
        end_pile = chrono::system_clock::now();
        cout << pairIndexImage.first << " ";
        cout << ((duration<double>)(end_pile-start_pipe)).count() << "s" << endl;

        /* push the image into display frame queue */
        mtxQueueShow.lock();
        queueShow.push(pairIndexImage);
        mtxQueueShow.unlock();
    }
}

/**
 * @brief Entry for running YOLO-v3 neural network for ADAS object detection
 *
 */
int main(const int argc, const char** argv) {

    if (argc == 1 || argc > 3 ) {
        cout << "Usage of this exe: ./yolo image_name[string]"   << endl;
        cout << "Usage of this exe: ./yolo image_name[string] i" << endl;
        cout << "Usage of this exe: ./yolo image_list[string] t" << endl;
        cout << "Usage of this exe: ./yolo image_list[string] p" << endl;
        return -1;
    }

    string model;
    if (argc == 2) {
        model = "i";
    } else {
        model = argv[2];
    }

    if(model == "p"){

        /* Attach to DPU driver and prepare for running */
        dpuOpen();

        /* Load DPU Kernels for YOLO-v3 network model */
        DPUKernel *kernel = dpuLoadKernel("yolo");

        /* Create n DPU Tasks for YOLO-v3 network model */
        vector<DPUTask *> task(NUM_YOLO_THREAD);
        generate(task.begin(), task.end(),
        std::bind(dpuCreateTask, kernel, 0));

        /* Spawn n+2 threads:
        - 1 thread for reading video frame
        - n identical threads for running YOLO-v3 network model
        - 1 thread for displaying frame in monitor
        */
        array<thread, (NUM_YOLO_THREAD+2)> threadsList = {
            thread(readFrame, argv[1]),
            thread(displayFrame),
            thread(runYOLO_video, task[0]),
            // thread(runYOLO_video, task[1]),
            // thread(runYOLO_video, task[2]),
            // thread(runYOLO_video, task[3]),
            // thread(runYOLO_video, task[4]),
            // thread(runYOLO_video, task[5]),
            // thread(runYOLO_video, task[6]),
            // thread(runYOLO_video, task[7]),
            // thread(runYOLO_video, task[8]),
            // thread(runYOLO_video, task[9]),
            // thread(runYOLO_video, task[10]),
            // thread(runYOLO_video, task[11]),
            // thread(runYOLO_video, task[12]),
            // thread(runYOLO_video, task[13]),
            // thread(runYOLO_video, task[14]),
            // thread(runYOLO_video, task[15]),
        };

        for (int i = 0; i < NUM_YOLO_THREAD+2; i++) {
            threadsList[i].join();
        }

        /* Destroy DPU Tasks & free resources */
        for_each(task.begin(), task.end(), dpuDestroyTask);

        /* Destroy DPU Kernels & free resources */
        dpuDestroyKernel(kernel);

        /* Dettach from DPU driver & free resources */
        dpuClose();

        // cout << "Calculating latency results..." << endl;
        // sort(in_pair.begin(), in_pair.end(), tpaircomp);
        // sort(out_pair.begin(), out_pair.end(), tpaircomp);
        // vector<timePair>::const_iterator in_it(in_pair.begin());
        // vector<timePair>::const_iterator in_ed(in_pair.end());
        // vector<timePair>::const_iterator out_it(out_pair.begin());
        // vector<timePair>::const_iterator out_ed(out_pair.end());
        // if(in_pair.size()!=out_pair.size()){
        //     cout << "vector size not match!!" << endl;
        // }
        // while(in_it!=in_ed){
        //     cout << in_it->first << " " << out_it->first << " ";
        //     cout << ((duration<double>)(out_it->second-in_it->second)).count() << "s" << endl;
        //     ++in_it;++out_it;
        // }
        
        return 0;

    } else if (model == "i") {

        start_time = chrono::system_clock::now();
        dpuOpen();
        end_time = chrono::system_clock::now();
        cout << "dpuOpen(): " << ((duration<double>)(end_time-start_time)).count() << "s" << endl;

        start_time = chrono::system_clock::now();
        Mat img = imread(argv[1]);
        end_time = chrono::system_clock::now();
        cout << "imread(): " << ((duration<double>)(end_time-start_time)).count() << "s" << endl;

        start_time = chrono::system_clock::now();
        DPUKernel *kernel = dpuLoadKernel("yolo");
        end_time = chrono::system_clock::now();
        cout << "dpuLoadKernel(): " << ((duration<double>)(end_time-start_time)).count() << "s" << endl;

        start_time = chrono::system_clock::now();
        DPUTask* task = dpuCreateTask(kernel, 0);
        end_time = chrono::system_clock::now();
        cout << "dpuCreateTask(): " << ((duration<double>)(end_time-start_time)).count() << "s" << endl;

        start_time = chrono::system_clock::now();
        runYOLO(task, img);
        end_time = chrono::system_clock::now();
        cout << "runYOLO(): " << ((duration<double>)(end_time-start_time)).count() << "s" << endl;

        imwrite("result.jpg", img);
        imshow("Xilinx DPU", img);
        waitKey(0);

        dpuDestroyTask(task);
        /* Destroy DPU Kernels & free resources */
        dpuDestroyKernel(kernel);

        /* Dettach from DPU driver & free resources */
        dpuClose();

        return 0;

    } else if (model == "t") {

        ifstream inputFile(argv[1]);
        vector<string> fileList;
        string line;

        if(!inputFile) {
            cout << "File list could not be opened\n";
            return -1;
        }

        while(getline(inputFile, line)) {
            fileList.push_back(line);
        }
        cout << "Number of files to be analyzed: " << fileList.size() << "\n";

        dpuOpen();
        DPUKernel *kernel = dpuLoadKernel("yolo");
        DPUTask* task = dpuCreateTask(kernel, 0);

        vector<string>::const_iterator it(fileList.begin());
        vector<string>::const_iterator end(fileList.end());
        for(;it != end;++it) {
            cout << "Start processing " << *it << endl;
            
            start_time = chrono::system_clock::now();
            Mat img = imread(it->c_str());
            end_time = chrono::system_clock::now();
            cout << "imread(): " << ((duration<double>)(end_time-start_time)).count() << "s" << endl;

            ofstream outfile;
            string fname = *it;
            fname.replace(fname.end()-4, fname.end(), ".txt");
            outfile.open(fname.c_str());
            cout << "Writing to file: " << fname.c_str() << endl;

            start_time = chrono::system_clock::now();
            /* mean values for YOLO-v3 */
            float mean[3] = {0.0f, 0.0f, 0.0f};
            int height = dpuGetInputTensorHeight(task, INPUT_NODE);
            int width = dpuGetInputTensorWidth(task, INPUT_NODE);
            setInputImageForYOLO(task, img, mean);
            end_time = chrono::system_clock::now();
            cout << "setInputImageForYOLO(): " << ((duration<double>)(end_time-start_time)).count() << "s" << endl;

            auto start_time = chrono::system_clock::now();
            dpuRunTask(task);
            auto end_dpu_time = chrono::system_clock::now();
            vector<vector<float>> res = postProcessResults(task, img, width, height);
            auto end_cpu_time = chrono::system_clock::now();
            duration<double> dpu_seconds = end_dpu_time - start_time;
            duration<double> cpu_seconds = end_cpu_time - end_dpu_time;
            cout << "DPU time: " << dpu_seconds.count() << "s" << endl;
            cout << "CPU time: " << cpu_seconds.count() << "s" << endl;

            float h = img.rows;
            float w = img.cols;
            for(size_t i = 0; i < res.size(); ++i) {
                float xmin = (res[i][0] - res[i][2]/2.0) * w + 1.0;
                float ymin = (res[i][1] - res[i][3]/2.0) * h + 1.0;
                float xmax = (res[i][0] + res[i][2]/2.0) * w + 1.0;
                float ymax = (res[i][1] + res[i][3]/2.0) * h + 1.0;


                if(res[i][res[i][4] + 6] > CONF ) {

                    // Results output
                    int type = res[i][4];  // result of classification
                    string type_names[7] = {"Broccoli","Cauliflower","Spectralon_15%", "Spectralon_30%",
                            "Spectralon_60%", "ColorChecker_C", "ColorChecker_Gy"}; // as in lady.names
                    cout << fixed << setprecision(5) << res[i][type + 6];
                    cout << "\t" << type_names[type] << "\t";
                    cout << xmin << " " << ymin << " " << xmax << " " << ymax << endl;

                    // outfile << type_names[type] << " ";
                    outfile << type_names[type] << " ";
                    outfile << res[i][type + 6] << " ";
                    outfile << xmin << " " << ymin << " " << xmax << " " << ymax << endl;

                    // Mark the detction results on the image with OpenCV
                    // Scalar(a,b,c)  We would be defining a BGR color such as: Blue=a, Green=b and Red=c
                    if (type==0) {          // Broccoli
                        rectangle(img, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0, 255, 0), 1, 1, 0);
                    } else if (type==1) {    // Cauliflower
                        rectangle(img, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(255, 255, 255), 1, 1, 0);
                    } else {                 // Others
                        rectangle(img, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0 ,255, 255), 1, 1, 0);
                    }
                }
            }
            outfile.close();

            // Save detection for review
            // fname.replace(fname.end()-4, fname.end(), ".jpg");
            // imwrite(fname.c_str(), img);

            // Show the images for inspection
            // imshow("Xilinx DPU", img);
            // waitKey(0);
        }

        dpuDestroyTask(task);
        /* Destroy DPU Kernels & free resources */
        dpuDestroyKernel(kernel);

        /* Dettach from DPU driver & free resources */
        dpuClose();

        return 0;

    } else {
        cout << "unknow type !"<<endl;
        cout << "Usage of this exe: ./yolo image_name[string]" << endl;
        cout << "Usage of this exe: ./yolo image_name[string] i" << endl;
        cout << "Usage of this exe: ./yolo image_list[string] t"<< endl;
        cout << "Usage of this exe: ./yolo video_name[string] v"<< endl;

        return -1;
    }


}
