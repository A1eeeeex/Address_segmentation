import torch
import numpy as np


def isset(v): 
    try : type (eval(v)) 
    except : return 0 
    else : return 1

if __name__ == "__main__":

    from transformers import AutoTokenizer, AutoModelForTokenClassification
    tokenizer = AutoTokenizer.from_pretrained("model\checkpoint-200")

    id2label = {
        0: "O",
        1: "T-B",
        2: "T-I",
        3: "P-B",
        4: "P-I",
        5: "A1-B",
        6: "A1-I",
        7: "A2-B",
        8: "A2-I",
        9: "A3-B",
        10: "A3-I",
        11: "A4-B",
        12: "A4-I",
    }
    label2id = {
        "O":0,
        "T-B":1,
        "T-I":2,
        "P-B":3,
        "P-I":4,
        "A1-B":5,
        "A1-I":6,
        "A2-B":7,
        "A2-I":8,
        "A3-B":9,
        "A3-I":10,
        "A4-B":11,
        "A4-I":12
    }
    model = AutoModelForTokenClassification.from_pretrained("model\checkpoint-200", id2label=id2label, label2id=label2id)
    
    while True:
        text = input('请输入地址数据: ')
        new_text = []
        for c in text:
            new_text.append(c)
        inputs = tokenizer(new_text, return_tensors="pt",is_split_into_words=True)

        with torch.no_grad():
            logits = model(**inputs).logits
        predictions = torch.argmax(logits, dim=2)
        predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
        predictions = np.array(predictions)[0]
        for i in range(1,len(predictions)-1):
            if predictions[i] == 1:
                tel = text[i-1]
            elif predictions[i] == 2:
                if not isset('tel'):
                    tel = text[i-1]
                else:
                    tel += text[i-1]
            elif predictions[i] == 3:
                per = text[i-1]
            elif predictions[i] == 4:
                if not isset('per'):
                    per = text[i-1]
                else:
                    per += text[i-1]
            elif predictions[i] == 5:
                add1 = text[i-1]
            elif predictions[i] == 6:
                if not isset('add1'):
                    add1 = text[i-1]
                else:
                    add1 += text[i-1]
            elif predictions[i] == 7:
                add2 = text[i-1]
            elif predictions[i] == 8:
                if not isset('add2'):
                    add2 = text[i-1]
                else:
                    add2 += text[i-1]
            elif predictions[i] == 9:
                add3 = text[i-1]
            elif predictions[i] == 10:
                if not isset('add3'):
                    add3 = text[i-1]
                else:
                    add3 += text[i-1]
            elif predictions[i] == 11:
                add4 = text[i-1]
            elif predictions[i] == 12:
                if not isset('add4'):
                    add4 = text[i-1]
                else:
                    add4 += text[i-1]

        if not isset('tel'):
            tel = '无'
        if not isset('per'):
            per = '无'
        if not isset('add1'):
            add1 = '无'
        if not isset('add2'):
            add2 = '无'
        if not isset('add3'):
            add3 = '无'
        if not isset('add4'):
            add4 = '无'

        res = {
            "电话":tel,
            "姓名":per,
            "一级地址":add1,
            "二级地址":add2,
            "三级地址":add3,
            "四级地址":add4
        }
        print('切分后的地址为：', res)
        del tel, per,add1,add2,add3,add4