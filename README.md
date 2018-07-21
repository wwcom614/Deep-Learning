学习深度学习领域知识时，边各种神经网络理论学习，边TensorFlow官网学习，边编码实践，循序渐进：TensorFlow编程基础和常用基本操作、可视化TensorBoard、线性模型+激活函数神经网络、卷积神经网络CNN、循环神经网络RNN之LSTM、谷歌图片识别模型InceptionV3直接应用看分类效果、基于自己的分类图片集在InceptionV3模型的pool3之上bottlenect训练自己的模型并看训练效果，基于自己的分类图片集基于slim里的InceptionV3模型训练自己的模型并看训练效果、生成随机验证码文本，基于ImageCaptcha生成验证码图片，并命名作为label，将这些图片转为tfrecord文件(protobuf格式)。学习标准的多任务学习以及tfrecord文件读写方式操作数据，熟悉使用input pipeline读取tfrecords文件，然后随机乱序，生成文件序列，读取并解码数据，输入模型训练的套路，调用alexnet_v2网络模型训练自己的模型，然后测试验证。


## FirstTry 
- HelloWorld.py：  
接触TensorFlow理论概念后，写的第一个程序。内容是尝试简单的常量和变量操作，主要用于理解TensorFlow的客户端模型和底层执行session层。  
尽管简单，在执行时居然遇到了问题(非代码和库原因)。花了几个小时排查解决，并将解决过程记录在自己的博客上[https://www.cnblogs.com/wwcom123/p/9251081.html])，分享给网上很多遇到同类问题但并未解决的同学们。

- ActivationFunction.py  
激活函数(Activation Function)：目的是向模型中加入非线性。实现tf.nn中几种常见的激活函数，绘图查看理解。

- FetchAndFeed.py  
代码实践：  
Fetch:在一个会话session里同时执行多个操作operation。 Feed：变量定义时用占位符，可以run的时候再Feed数据，以字典形式传入。 

- LR_Using_GD.py  
用梯度下降法求解线性回归问题：
构造加噪声的线性数据作为数据集Orignal Data，客户端构造线性回归模型预测以及定义了损失函数，使用了tf.train中的基于梯度下降法的优化器，让损失函数最小。  
调用tf.Session执行：先初始化所有变量，循环执行30次拟合fit，matplot绘图查看拟合情况。

## Linear
-  LinearActivationRegression.py  
基于自己构造的加噪声x平方函数数据，组建1层隐藏层的线性模型+激活函数神经网络，梯度下降求解最小损失函数，解决回归问题。画图查看拟合效果。

-  Linear_Mnist.py  
加深理解，再次尝试，这次基于手写数字识别数据集，组建3层线性模型+激活函数神经网，学习多种优化器原理，然后尝试使用adam等多种优化器最小化损失函数。输出每个迭代测试准确率等。同时，输出并保存模型、参数、准确率和损失变化到TensorBoard中，使用TensorBoard查看。


## TensorBoard
- TensorBoardFirstTry.py  
首次接触TensorBoard，尝试使用tf.summary生成graph文件，用tensorboard查看，理解TensorFlow的各种节点、张量、流图。

-  VarAnalysis.py  
写了工具函数，用于对一个矩阵或向量，在TensorBoard中的scalar标签查看网络运行状态：均值变化、标准差变化、最大值变化、最小值变化、直方图等。

-  InceptionV3TensorBoard.py  
读取谷歌的InceptionV3的训练模型文件classify_image_graph_def.pb，在TensorFlow中查看其内部构造--像个bottle，注意pool3位置，像bottleneck，见下面的InceptionV3章节描述。

## CNN
线性模型每次都是全连接计算，权值和偏置值随样本数增大呈笛卡尔积式增长，计算量过大，而且每次全连接有可能过拟合。卷积神经网络基于局部感受野的生物实践理论，使用卷积核采样+池化亚采样方式，减少前期计算量，同时卷积可以局部平滑，实践证明效果更好。
-  CNN_Layer_Conv2d_Mnist.py  
学习了CNN神经网络理论后，自己动手cnn卷积神经网络编程，拟合手写数字识别数据集，使用的是tf.layers.conv2d。2层卷积+池化，输出准确率和预测结果对比。

-  CNN_NN_Conv2d_Mnist.py  
觉得对CNN神经网络理解还不到位，知乎上学习了各路大神对卷积理论的解释和应用举例，自己动手再次cnn卷积神经网络编程，拟合手写数字识别数据集，本次改为使用更底层些的tf.nn.conv2d。2层卷积+max池化,2层全连接，同时学习了交叉熵代价函数相对二次代价函数更为使用这种非线性网络理论，引入使用，预测并输出准确率。同时，输出并保存模型、参数、准确率和损失变化到TensorBoard中，使用TensorBoard查看。

## RNN
循环神经网络，或者叫递归神经网络，会将上一个输出作为下一个的输入辅助决策，例如在一段时间内的语音、视频截图，一句话的文本分词等。
-  LSTM_Mnist.py   
LSTM(Long Short Term Memory)是一种RNN算法，使用input/forget/output三种门避免RNN的梯度消失问题，可以控制信号向后传多少个，影响哪个输出。结合上述理论实践LSTMCell，并梳理其各个参数与理论的对应关系。预测并输出准确率。  
同时，练习了如何将训练好的模型保存到文件，以及从文件加载模型到TensorFlow。

## IncepetionV3
IncepetionV3是谷歌参加ImageNet大赛提供的经典的图片预测模型。其内部多层的卷积和池化，层数增加模型深度
多次卷积和池化后，有个特殊的mix，增加了模型的宽度。
###  DirectApply
打算直接使用IncepetionV3模型，引入图片看分类效果
- Id_Desc_MapToFile.py  
谷歌的模型输出结果是2个文件，查看预测结果时不方便，自己写了个代码，将这2个文件转换成1个字典文件(key:预测编号，value：预测编号对应的英文描述)。方便最终预测结果查看调用。执行后生成文件id_desc.txt

-  ImagePredictByIncepetionV3.py  
创建一个图引入google训练好的模型classify_image_graph_def.pb，从网上下载了几张猫狗飞机之类的图片让其预测，预测结果使用id_desc.txt翻译成语言描述查看。

###  Pool3Train:  
基于自己的图片，微调参数训练InceptionV3模型后预测：前端卷积保持，后端在pool3之后(bottle neck之上)训练调优。  
相对全新训练模型的好处：1.训练速度快；2.迭代次数少(200次和几十上百万次的差别)；3.所需样本少
-  retrain.py：  
谷歌在tensorflow-r1.7\tensorflow\examples\image_retraining\retrain.py，该文件用于只训练IncepetionV3模型最上方(pool3上方，bottleneck)参数。 
- retrain.bat：  
该批处理文件，带参数调用retrain.py。参数就是我们要配置的：  
bottleneck_dir：要提前建立一个文件夹，用途见训练完成说明；  
how_many_training_steps：训练次数；  
model_dir：谷歌InceptionV3模型classify_image_graph_def.pb所在路径文件夹； 
output_graph：输出训练好的模型到该文件夹；  
output_labels：输出训练数据标签到该文件夹；  
image_dir：待训练图片集，每种分类图片一个文件夹，路径名称全英文小写。 

-  训练完成后  
会在bottleneck_dir目录下，分image_dir下的分类目录名存放，生成每张图片的InceptionV3模型pool3之下的所有参数，记录在“图片名称_inception_v3.txt”文件内；  
会在output_graph目录下，生成基于自己的图片训练的新模型：output_graph.pb；  
会在output_labels目录下，生成分类labels：output_labels.txt;
会在所在盘符根目录下生成一个tmp文件夹，再次训练需要手工删除tmp文件夹中临时文件。

- ImagesTest.py  
自己编写的测试结果代码：读取训练结果labels文件output_labels.txt，转换成行id和名称的键值对字典，封装成函数id_to_name_func；创建一个图，引入自己训练好的模型output_graph.pb；服务端session执行：读取测试图片放入模型预测，取预测结果最大的labelid输出，预测结果调用id_to_name_func转换成label名称，同时计算准确率score打印。

###  SlimTrain:  
基于自己的图片，全新训练InceptionV3模型后预测。  
全新训练模型：1.训练速度慢；2.迭代次数多(几十上百万次)；3.所需样本多
- Images_to_tfrecords.py  
首先，准备训练样本图片分目录分类存：每个目录名是分类名称，该目录内存放该类的所有样本图片。  
然后，获取图片全路径+文件名，以及分类名称(目录名称)。遍历，把所有图片转为标准格式的tfrecord文件。输出labels文件。
-  train.bat  
下载slim：https://github.com/tensorflow/models/research/slim, preprocessing、nets、datasets、deployment、Read_tfrecords_to_Memory.py、train_image_classifier.py做相应修改，使用train.bat参数调用训练。


## Captcha_MultiTask  
生成随机验证码文本，基于ImageCaptcha生成验证码图片，并命名作为label，将这些图片转为tfrecord文件(protobuf格式)。学习标准的多任务学习以及tfrecord文件读写方式操作数据，熟悉使用input pipeline读取tfrecords文件，然后随机乱序，生成文件序列，读取并解码数据，输入模型训练的套路，调用alexnet_v2网络模型训练自己的模型，然后测试验证。  

-  Gen_CaptchaImages.py  
生成随机验证码文本，使用captcha.image的ImageCaptcha，将文本转换为随机验证码图片，图片的名称使用验证码文本名(做label用)，作为数据源。

-  CaptchaImages_to_tfrecords.py  
将验证码图片转为2个tfrecord文件(训练集文件和测试集文件)，同时获取图片的名称作为label

-  Captcha_Train.py  
读取tfrecord文件放入queue，然后随机乱序，生成文件序列，读取并解码数据，输入模型训练。  

1.用tf.train.string_input_producer读取tfrecords文件的到队列queue中，输入参数num_epoches表示每次读取图片个数,输入参数shuffle表示是否打乱后读取，默认打乱后读取。   
2.使用tf.TFRecordReader读取queue中的tfrecords文件，之后通过一个解析器tf.parse_single_example。label直接就是字典了，图片还需要用解码器tf.decode_raw继续解码。  
3.使用tf.train.shuffle_batch函数把一个个小样本的tensor打包成一个高一维度的样本batch，输入是单个样本，输出就是4D的样本batch了，其内部原理是创建一个queue，然后不断调用单样本tensor获得样本，直到queue里边有足够的样本，然后一次返回一堆样本，组成样本batch。  
4.session创建多线程多任务从queue中批量batch获取数据，调用alexnet_v2模型，损失函数是4位验证码交叉熵之和，学习过程中每2000次减小一次学习率，使用adam最小化损失函数，进行模型训练，最终输出训练好的模型以及准确率和损失偏差。

-  Captcha_Test.py  
与Captcha_Train.py类似，不同的是读取Captcha_Train.py训练好的模型，读取测试集tfrecords文件做测试验证，输出图片、真实值、预测值对比看分类效果。

- 调用和修改alexnet_v2模型说明  
slim：https://github.com/tensorflow/models/research/slim/nets/下有nets_factory.py和alexnet.py，拷贝到自己的Captcha_MultiTask/nets包中。修改nets_factory.py包：只使用alexnet_v2模型；修改alexnet.py：主要是最后的net输出，4位验证码需要使用4个net。