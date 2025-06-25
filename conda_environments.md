## 简单创建conda模型训练环境

```bash
conda create -n py311_torch python==3.11.12

# for cuda 12.6 nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

本项目使用的是Rapids环境，Rapids版本为25.04，Pytorch版本为2.7.1，CUDA版本为12.6.


## 创建Rapids环境

### 1. 安装Miniconda
清华源下载：https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/
```bash
sudo su
bash /data/Miniconda3-latest-Linux-x86_64.sh 
```
一路yes或者回车就行了

### 2. 激活conda
```bash
source ~/miniconda3/bin/activate #激活conda
conda init    #初始化conda
```
主机名前面出现(base)则激活成功，也可以which conda看路径

### 3. 配置镜像源
```bash
conda config --show # 查看当前源
```
- 添加镜像源
```bash
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes #设置搜索时显示通道地址
```

### 4. 下载RAPIDS环境
RAPIDS官网：https://docs.rapids.ai/install#selector
根据实际环境得到安装命令
例如我这里的：
```bash
conda create -n rapids-25.04 -c rapidsai -c conda-forge -c nvidia  \
    rapids=25.04 python=3.11.12 'cuda-version>=12.0,<=12.8'
```
执行命令，创建对应的conda环境，一路回车即可

### 5. 配置Jupyter Lab
- 安装Jupyter Lab
```bash
conda install -c conda-forge jupyterlab 
```
- 使用命令创建配置文件
输入以下命令后，会生成/home/用户名/.jupyter/jupyter_notebook_config.py配置文件
```bash
jupyter notebook --generate-config --allow-root
```
- 修改配置文件
```bash
ipython
```
- 在In[1]:后面输入
```bash
from notebook.auth import passwd
```
- 在In[2]:后面输入
```bash
passwd()
```
输入密码后再次输入密码确认一遍，之后会生成秘钥

将Out[2]输出的密钥保存下来，稍后会用到。然后输入以下命令退出ipython环境
```bash
exit
```
接着在shell中输入
```bash
vi /root/.jupyter/jupyter_notebook_config.py
```
修改配置文件 jupyter_lab_config.py，在配置文件中添加以下内容：
```bash
c.NotebookApp.ip = '*'        # 所有ip均可访问，如果指定ip的话，则需要输入具体的ip地址
c.NotebookApp.password = '刚才保存的密钥'
c.NotebookApp.open_browser = False   # 运行Jupyter notebook后是否自动打开浏览器
c.NotebookApp.port = 8888            # 端口号
c.NotebookApp.notebook_dir = '/home' # ipython文件保存的路径，按自己的路径修改，必须要配置，不然访问时>会出现404错误。
```
### 6. 安装PyTorch和Transformers
```bash
# for cuda 12.6 nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install transformers=4.52.4 nlp kagglehub datasets==3.6.0 evaluate==0.4.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
安装完记得确认pip list，看一下torch是否为cuda版本
例如我这里是：
```bash
torch                     2.7.1+cu126
torchaudio                2.7.1+cu126
torchvision               0.22.1+cu126
```
如果不是，需要在官网查找对应的安装指令：https://pytorch.org/get-started/locally/

### 7. 启动Jupyter Lab
```bash
cd BERT_Learning
conda activate rapids-25.04
jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser
```
