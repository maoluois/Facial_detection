## homework 00
## 修改gradio窗口布局 
修改gradio窗口布局，完成全部作业后发布公网链接让朋友试用。
## 增加AR demo
https://docs.opencv.org/4.x/dc/d2c/tutorial_real_time_pose.html 
参考上面链接，修改代码。实现gdut校徽，贴人头上。有AR仿射变换效果。 
## 调用增加新的facial landmark detection模型
增加一个模型 Torchlm人脸检测库 :https://github.com/DefTruth/torchlm 
新建一个tab： “models comparison”页。
要求UI为4窗口：1原图视频，1个原视频上叠加mediapipe68点， 1个叠加Torchlm 68点， 1个叠加2个模型的对应点连线（线的长度反应了2个模型的定位差异）
