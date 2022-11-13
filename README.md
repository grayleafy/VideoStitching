# VideoStitching
# 双路高清视频实时拼接系统
使用OpenCV和CUDA构建的双路视频拼接系统，SIFT算法初始化单应性矩阵，每一帧曝光补偿、计算最佳缝合线、渐入渐出融合。通过线程池对多帧视频图像并行拼接。还在完善中。

对于固定摄像头的场景，1920 * 1080的两幅视频可以达到40FPS的拼接速度。
![72`AYE1_O88PEUTJ@P7)UG4](https://user-images.githubusercontent.com/86156654/201525662-e36cb10d-9813-470a-b1c5-f8e44151f18e.png)

## 拼接效果

![image](https://user-images.githubusercontent.com/86156654/201525816-7d9e9d34-0b62-4242-8361-51d0418b6884.png)
![image](https://user-images.githubusercontent.com/86156654/201525815-9346a741-9a62-406a-8ab8-30d5706543f2.png)


![image](https://user-images.githubusercontent.com/86156654/201525854-4cc0b588-3e5f-49bb-90ac-5449172a2849.png)

