# Connectionist Temporal Classification(CTC)  
 
## Files  
configs.py(参数),data_make.py(产生数据),data_utils.py(处理数据),model.py(BiLSTM-CTC 模型),predict.py(测试数据)  
  
## Train Data  
![](https://github.com/Sunsapience/RNN_CTC/blob/master/show/Figure_.png)  
  
## Test Result  
![](https://github.com/Sunsapience/RNN_CTC/blob/master/show/Figure_1-1.png)  
![](https://github.com/Sunsapience/RNN_CTC/blob/master/show/Figure_1-2.png)  
  
## CTC_LOSS
ctc_loss主要解决的序列模型输出对齐问题。在语音识别、文字识别中，常会出现序列模型(RNN)输出长度与标签长度不一致问题，而CTC可以解决这一问题。其核心是在标签中添加'blank'字符，改变原始的标签序列，运用动态规划算法算法寻找最佳路径。路径选择结果构成 ctc_loss。    
通常情况下，RNN可以实现 N-N，N-1，0/1-N，但很难实现N-M(seq2seq用两个RNN实现了自适应长度)。带有ctc的RNN可以实现N-M(一般N>M)，这使得RNN的应用场景更加广泛。  
  
## 参考  
[1] [Connectionist Temporal Classification : Labelling Unsegmented Sequence Data with Recurrent Neural Networks.](https://dblp.uni-trier.de/db/conf/icml/icml2006.html)  
[2] [博客1](https://xiaodu.io/ctc-explained/)  
[3] [博客2](https://blog.csdn.net/JackyTintin/article/details/79425866)  
[4] [参考代码](https://github.com/stardut/ctc-ocr-tensorflow)

