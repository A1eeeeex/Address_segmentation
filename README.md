# 文本地址切分
- > 时间: 2023/02/08
  
## 描述
- 描述: 提供了地址信息文本的切分能力。
- 输入：一段包含电话，姓名和地址的中文文本，顺序任意，如：
```
  12345678910马拉丁达广东省广州市海珠区阅江西路222号(广州塔地铁站B口步行160米)
```
- 输出：该文本的切分词典
```
{   '电话': '12345678910', 
    '姓名': '马拉丁达', 
    '一级地址': '广东省', 
    '二级地址': '广州市', 
    '三级地址': '海珠区', 
    '四级地址': '阅江西路222号(广州塔地铁站B口步行160米)'}
```

## 实现原理
在bert-base-chinese模型上进行了微调(fine-tune)，地址切分任务可以看作是特殊的命名实体识别(NER)任务，本任务使用的标签如下
```
label_list = [
        "O", 
        "T-B",  # 电话的开头
        "T-I",  # 电话的中间部分
        "P-B",  # 姓名的开头
        "P-I",  # 姓名的中间部分
        "A1-B", # 一级地址的开头
        "A1-I", # 一级地址的中间部分
        "A2-B", # ...
        "A2-I",
        "A3-B",
        "A3-I",
        "A4-B",
        "A4-I"
        ]
```

## 数据集
数据集来自飞桨社区文章《序列标注任务，如何从快递单中抽取关键信息》：https://aistudio.baidu.com/aistudio/projectdetail/131360?channel=0&channelType=0&sUid=2818587&shared=1&ts=1675839980817

共有1600条训练数据，200条验证数据和200条测试数据。

## 环境
- transformers
- datasets
- numpy
- torch
## 模型训练
```python
python train.py
```
模型训练代码参考了https://huggingface.co/docs/transformers/tasks/token_classification#token-classification
共训练了2个epochs，学习率为2e-5，训练后的评估结果如下

|loss|precision|recall|f1|accuracy|
|-----|-----|-----|-----|-----|
|0.0123|0.9937|0.9937|0.9937|0.9981|

模型取得了非常优秀的效果

## 模型推测
```
python predict.py
```
运行predict文件，按提示输入句子，按ctrl+c退出
