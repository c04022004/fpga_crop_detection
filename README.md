# fpga_crop_detection

Part of the program used in the paper [Vision Guided Crop Detection in Field Robots using FPGA-Based Reconfigurable Computers](https://ieeexplore.ieee.org/document/9181302)

The backbone of the program is forked from `Xilinx/Edge-AI-Platform-Tutorials` (now part of `Xilinx/Vitis-In-Depth-Tutorial`)  
The closest code to the original source should be [this tutorial](https://github.com/Xilinx/Vitis-In-Depth-Tutorial/blob/master/Machine_Learning/Design_Tutorials/07-yolov4-tutorial/README.md),
but with the orignal code written with the use of YOLOv3.

To repurpose the porgram to do some benchmark and pushing for higher throughput, I've adapted the source code to run a modified version (thru transfer learning) of the YOLOv3 (with only 7 classes).
As the loading and resizing of the image from external SD card soon becomes the bottleneck, changes have been made to preload images to RAM and pipelined the resizing & inference process.

The hardware used in the project:  
<img alt="Hardware" src="https://github.com/c04022004/fpga_crop_detection/blob/master/img/fpga_running.jpg" width="700">

Some of the screenshots are listed as below:  
<img alt="Detection" src="https://github.com/c04022004/fpga_crop_detection/blob/master/img/ok_c.jpeg" width="700">
<img alt="DPU Usage (1 core)" src="https://github.com/c04022004/fpga_crop_detection/blob/master/img/dpu1.png" width="700">
<img alt="DPU Usage (3 core)" src="https://github.com/c04022004/fpga_crop_detection/blob/master/img/dpu3.png" width="700">
