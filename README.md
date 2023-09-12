# **SMILES Generation with CVAE**

**此项目使用条件变分自动编码器 (CVAE) 来预测生成SMILES字符串。**

# 文件描述
model.py - 包含CVAE的PyTorch实现。
train.py - 包含一个修正版的CVAE模型，进行模型训练的相关代码。
sample.py - 包含模拟采样和序列转换的占位符函数。
utils.py - 一些用于SMILES字符串转换和处理的实用工具函数。
egfr_smiles.txt - 包含了多个代表化合物结构的SMILES字符串。
# 如何使用
安装所有必要的依赖，包括PyTorch和rdkit。
使用train.py训练模型。
使用sample.py从训练好的模型中采样新的SMILES字符串。
使用utils.py中的工具函数进行SMILES字符串的转换和处理。
# 注意
sample.py中的函数是占位符，可能需要进一步的实现。
在使用此代码进行训练或采样之前，请确保理解并适当调整所有参数和设置。
