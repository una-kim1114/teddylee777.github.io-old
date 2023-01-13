---
layout: page
title: "[huggingface] 허깅페이스 사전학습 토크나이저와 모델로 텍스트 분류(Text Classification)하기"
description: "허깅페이스 사전학습 토크나이저와 모델로 텍스트 분류(Text Classification)에 대해 알아보겠습니다."
headline: "허깅페이스 사전학습 토크나이저와 모델로 텍스트 분류(Text Classification)에 대해 알아보겠습니다."
categories: hugging face
tags: [python, 파이썬, huggingface, 허깅페이스, 텍스트 분류, 자연어처리, 감정 분석, 자연어처리 사전학습, bert, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터, 테디노트]
comments: true
published: true
typora-copy-images-to: ../images/2023-01-10
---



요즘에 자연어 처리(특히나 트랜스포머 계열의 모델)를 공부한다면 절대적인 입지를 가지고 있는 [허깅페이스(Hugging Face)](https://huggingface.co/)를 빼놓고 얘기할 수는 없을 것입니다. 하지만, 이제 더이상 자연어 처리 도메인에 국한되지 않고 비전(Vision), 오디오(Audio) 등을 포함한 범위도 넓혀 가고 있습니다. 

허깅페이스는 사전 학습(Pre-trained)된 모델과 학습 스크립트(Trainer)등을 제공하는 플랫폼입니다. 특히나 자연어 처리에서 사용되는 사전 학습된 토크나이저(Tokenizer)도 손쉽게 다운로드 받아 활용할 수 있도록 그 발판을 마련해 주고 있습니다.

![Hugging Face](../images/2023-01-10/hugginface.png)

허깅페이스가 많은 이들로부터 사랑받게 된 가장 큰 이유 중 하나는 **코드 몇 줄 만으로 사전 학습 모델을 다운로드 받아 미세조정(Fine Tuning)을 거쳐 성능 좋은 모델을 손쉽게 완성할 수 있다는 점**입니다. 통합된 인터페이스를 제공하기 때문에 거의 동일한 함수를 사용하지만 내가 원하는 모델의 명칭을 지정하거나 바꾸기만 하면 손쉽게 가져다 쓸 수 있습니다.



예를 들면 아래의 예시 코드에서,

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
```

문자열로 지정한 부분 즉 **사전 학습 모델의 명칭**만 바꿔주면 되는겁니다.



아래의 튜토리얼은 텍스트 분류를 허깅페이스의 사전 학습 모델과 토크나이저를 로드하고, `Trainer`를 활용하여 미세조정(Fine Tuning) 하는 방법에 대해 다룹니다. 가장 심플한 방법으로 학습하고 예측까지 수행합니다. 추후에 세분화된 기능과 커스텀 기능에 대해 다뤄볼 예정입니다.






				HTML


​					
​				
​				
​						
​				
​			
​		<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

```
table.dataframe th {
  text-align: center;
  font-weight: bold;
  padding: 8px;
}

table.dataframe td {
  text-align: center;
  padding: 8px;
}

table.dataframe tr:hover {
  background: #b8d1f3; 
}

.output_prompt {
  overflow: auto;
  font-size: 0.9rem;
  line-height: 1.45;
  border-radius: 0.3rem;
  -webkit-overflow-scrolling: touch;
  padding: 0.8rem;
  margin-top: 0;
  margin-bottom: 15px;
  font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
  color: $code-text-color;
  border: solid 1px $border-color;
  border-radius: 0.3rem;
  word-break: normal;
  white-space: pre;
}
```

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>

## 데이터셋 다운로드



`sarcasm.json` 데이터셋을 다운로드 받습니다.



```python
import urllib
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

SEED = 123

# 데이터셋 다운로드
url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
urllib.request.urlretrieve(url, 'sarcasm.json')

# JSON 파일을 데이터프레임으로 로드
df = pd.read_json('sarcasm.json')
df = df.rename(columns={
    'headline': 'sentence', 
    'is_sarcastic': 'label'
})
df
```


				HTML


​					
​				
​				
​						
​				
​			
​		
​

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>


				HTML


​					
​				
​				
​						
​				
​			
​		


​      
​      article_link
​      sentence
​      label

  


​    
​      0
​      https://www.huffingtonpost.com/entry/versace-b...
​      former versace store clerk sues over secret 'b...
​      0


​    
​      1
​      https://www.huffingtonpost.com/entry/roseanne-...
​      the 'roseanne' revival catches up to our thorn...
​      0


​    
​      2
​      https://local.theonion.com/mom-starting-to-fea...
​      mom starting to fear son's web series closest ...
​      1


​    
​      3
​      https://politics.theonion.com/boehner-just-wan...
​      boehner just wants wife to listen, not come up...
​      1


​    
​      4
​      https://www.huffingtonpost.com/entry/jk-rowlin...
​      j.k. rowling wishes snape happy birthday in th...
​      0


​    
​      ...
​      ...
​      ...
​      ...


​    
​      26704
​      https://www.huffingtonpost.com/entry/american-...
​      american politics in moral free-fall
​      0


​    
​      26705
​      https://www.huffingtonpost.com/entry/americas-...
​      america's best 20 hikes
​      0


​    
​      26706
​      https://www.huffingtonpost.com/entry/reparatio...
​      reparations and obama
​      0


​    
​      26707
​      https://www.huffingtonpost.com/entry/israeli-b...
​      israeli ban targeting boycott supporters raise...
​      0


​    
​      26708
​      https://www.huffingtonpost.com/entry/gourmet-g...
​      gourmet gifts for the foodie 2014
​      0

  


26709 rows × 3 columns​

</div>

## 데이터셋 분할



```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, random_state=SEED)
```

```python
# train 데이터셋 출력
train.head()
```


				HTML


​					
​				
​				
​						
​				
​			
​		
​

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>


				HTML


​					
​				
​				
​						
​				
​			
​		


​      
​      article_link
​      sentence
​      label

  


​    
​      7917
​      https://www.theonion.com/disturbance-of-arafat...
​      disturbance of arafat's grave casts horrible c...
​      1


​    
​      23206
​      https://www.huffingtonpost.com/entry/15-photos...
​      15 photos of hot dudes supporting bernie sande...
​      0


​    
​      4611
​      https://www.huffingtonpost.com/entry/illinois-...
​      6 things you need to know about the nation's s...
​      0


​    
​      11937
​      https://local.theonion.com/really-ugly-shark-t...
​      really ugly shark tired of being mistaken for ...
​      1


​    
​      9334
​      https://local.theonion.com/friends-wife-encoun...
​      friend's wife encountered twice a year
​      1

  



</div>



```python
# test 데이터셋 출력
test.head()
```


				HTML


​					
​				
​				
​						
​				
​			
​		
​

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>


				HTML


​					
​				
​				
​						
​				
​			
​		


​      
​      article_link
​      sentence
​      label

  


​    
​      22288
​      https://www.huffingtonpost.com/entry/steve-wil...
​      steve wilson on 'the making of gone with the w...
​      0


​    
​      16228
​      https://local.theonion.com/standards-lowered-f...
​      standards lowered for second search through fr...
​      1


​    
​      4905
​      https://www.huffingtonpost.comhttp://www.thede...
​      surgical tech in needle-swap scandal at swedis...
​      0


​    
​      8947
​      https://www.huffingtonpost.com/entry/donald-tr...
​      ferguson is not among the most dangerous place...
​      0


​    
​      3706
​      https://politics.theonion.com/bill-clinton-res...
​      bill clinton resting up to sit upright at next...
​      1

  



</div>

## 토큰화가 적용된 데이터셋



```python
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class TokenDataset(Dataset):
  
    def __init__(self, dataframe, tokenizer_pretrained):
        # sentence, label 컬럼으로 구성된 데이터프레임 전달
        self.data = dataframe        
        # Huggingface 토크나이저 생성
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained)
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['sentence']
        label = self.data.iloc[idx]['label']

        # 토큰화 처리
        tokens = self.tokenizer(
            sentence,                # 1개 문장 
            return_tensors='pt',     # 텐서로 반환
            truncation=True,         # 잘라내기 적용
            padding='max_length',    # 패딩 적용
            add_special_tokens=True  # 스페셜 토큰 적용
        )

        input_ids = tokens['input_ids'].squeeze(0)           # 2D -> 1D
        attention_mask = tokens['attention_mask'].squeeze(0) # 2D -> 1D

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask, 
            'label': torch.tensor(label)
        }
```

데이터셋 인스턴스 생성



```python
# distilbert-base-uncased 토크나이저 지정
tokenizer_pretrained = 'distilbert-base-uncased'

# train, test 데이터셋 생성
train_data = TokenDataset(train, tokenizer_pretrained)
test_data = TokenDataset(test, tokenizer_pretrained)
```

## Model

device 를 지정합니다 (`'cuda'` or `'cpu'`).

```python
import torch

# device 지정
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
```


				HTML


​					
​				
​				
​						
​				
​			
​		cuda:1




`TrainingArguments` 에 학습에 적용할 몇 가지 옵션을 다음과 같이 지정합니다. 세부 옵션에 대한 내용은 주석으로 달아 놓았습니다.

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments


# Fine-Tuning을 위한 옵션 지정
training_args = TrainingArguments(
    output_dir='./results',          # 결과 값이 저장될 디렉토리 지정
    num_train_epochs=3,              # 학습 epoch
    per_device_train_batch_size=16,  # training 배치사이즈
    per_device_eval_batch_size=64,   # evaluation 배치사이즈
    warmup_steps=500,                # leaning rate 스케줄러의 웜업 step
    weight_decay=0.01,               # weight decay 강도
    logging_dir='./logs',            # 로그를 저장할 디렉토리
    logging_steps=200,               # 로그 출력 step
)
```



사전 학습 모델을 지정합니다. 아래의 예제에서는 `'distilbert-base-uncased'` 모델을 가져왔습니다.

원하는 모델은 [Hugging Face Models](https://huggingface.co/models) 에서 검색해 볼 수 있습니다. 왼쪽에 필터링 조건에 한국어, 라이브러리(PyTorch/TensorFlow) 등을 지정할 수 있습니다. 한국어 텍스트 분류기를 위한 사전 학습 모델은 검색어에 `kor`을 입력해 보시기 바랍니다.

사전 학습 모델을 다운로드 받은 후 `Trainer`를 활용하여 학습을 손쉽게 진행합니다.

```python
# pre-trained 모델 지정
model_pretrained = 'distilbert-base-uncased'

# 모델 다운로드, num_labels 지정, device 지정
model = AutoModelForSequenceClassification.from_pretrained(model_pretrained, num_labels=2).to(device)

# Trainer 생성 후, model, train, test 데이터셋 지정
trainer = Trainer(
    model=model,                     # 이전에 불러온 허깅페이스 pretrained 모델
    args=training_args,              # 이전에 정의한 training arguments 지정
    train_dataset=train_data,        # training 데이터
    eval_dataset=test_data           # test 데이터
)

# trainer 를 활용한 학습 시작
trainer.train()
```


				HTML


​					
​				
​				
​						
​				
​			
​		Saving model checkpoint to ./results/checkpoint-500
Configuration saved in ./results/checkpoint-500/config.json
Model weights saved in ./results/checkpoint-500/pytorch_model.bin
Saving model checkpoint to ./results/checkpoint-1000
Configuration saved in ./results/checkpoint-1000/config.json
Model weights saved in ./results/checkpoint-1000/pytorch_model.bin
Saving model checkpoint to ./results/checkpoint-1500
Configuration saved in ./results/checkpoint-1500/config.json
Model weights saved in ./results/checkpoint-1500/pytorch_model.bin​

Training completed. Do not forget to share your model on huggingface.co/models =)

TrainOutput(global_step=1878, training_loss=0.1813284194253631, metrics={'train_runtime': 445.8676, 'train_samples_per_second': 134.778, 'train_steps_per_second': 4.212, 'total_flos': 7960363387435008.0, 'train_loss': 0.1813284194253631, 'epoch': 3.0})
</pre>



학습이 완료된 후 `trainer`의 `predict()` 함수에 `test_data` 를 전달하여 inference 를 수행합니다.

```python
# 학습된 trainer로 예측
predictions = trainer.predict(test_data)
predictions
```


				HTML


​					
​				
​				
​						
​				
​			
​		***** Running Prediction *****
  Num examples = 6678
  Batch size = 128
​




				HTML


​					
​				
​				
​						
​				
​			
​		PredictionOutput(predictions=array([[ 2.9732733, -2.9471958],
​	   [-4.0222363,  3.6413522],
​	   [ 3.8347576, -3.318453 ],
​	   ...,
​	   [ 2.824299 , -2.4794154],
​	   [ 3.5981152, -3.2576218],
​	   [ 4.025952 , -3.6779523]], dtype=float32), label_ids=array([0, 1, 0, ..., 0, 0, 0]), metrics={'test_loss': 0.3030776381492615, 'test_runtime': 13.6168, 'test_samples_per_second': 490.424, 'test_steps_per_second': 3.892})




예측된 결과는 `label_ids` 에 담겨 있습니다.

```python
# 예측 결과는 label_ids 에 담겨 있음
predictions.label_ids
```


				HTML


​					
​				
​				
​						
​				
​			
​		array([0, 1, 0, ..., 0, 0, 0])




정확도 평가를 수행합니다. `test['label']` 값과 비교하면 정확도가 산출됩니다.

```python
# 평가
accuracy = (test['label'] == predictions.label_ids).mean()
accuracy
```


				HTML


​					
​				
​				
​						
​				
​			
​		1.0

