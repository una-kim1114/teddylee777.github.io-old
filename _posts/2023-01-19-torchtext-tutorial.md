---
layout: page
title: "torchtext를 활용한 텍스트 데이터 전처리 방법"
description: "torchtext를 활용한 텍스트 데이터 전처리 방법에 대해 알아보겠습니다."
headline: "torchtext를 활용한 텍스트 데이터 전처리 방법에 대해 알아보겠습니다."
categories: pytorch
tags: [python, 파이썬, torchtext, pytorch, 파이토치, 자연어 처리, 텍스트 전처리, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터, 테디노트]
comments: true
published: true
typora-copy-images-to: ../images/2023-01-18

---

`torchtext`는 pytorch 모델에 주입하기 위한 텍스트 데이터셋을 구성하기 편하게 만들어 주는 데이터 로더(Data Loader) 입니다. `torchtext` 를 활용하여 CSV, TSV, JSON 등의 정형 데이터셋을 쉽게 로드하도록 도와주는 `TabularDataset` 클래스의 활용 방법과 제공해주는 토크나이저(tokenizer) 워드 벡터(Word Vector) 를 적용하는 방법에 대하여 알아보겠습니다.

튜토리얼의 끝 부분에는 Pandas 의 DataFrame을 Data Loader 로 쉽게 변환하는 방법도 알아보도록 하겠습니다.

예제 코드는 아래에서 확인할 수 있습니다.




				HTML


​					
​				
​				
​						
​				
​			
		<head>
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
​
table.dataframe td {
  text-align: center;
  padding: 8px;
}
​
table.dataframe tr:hover {
  background: #b8d1f3; 
}
​
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

# torchtext 튜토리얼

## 샘플 데이터셋 다운로드



```python
import urllib

url = 'https://storage.googleapis.com/download.tensorflow.org/data/bbc-text.csv'
urllib.request.urlretrieve(url, 'bbc-text.csv')
```


				HTML


​					
​				
​				
​						
​				
​			
		('bbc-text.csv', )


Pandas로 데이터 로드 및 출력



```python
import pandas as pd

df = pd.read_csv('bbc-text.csv')
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
      category
      text

  


​    
      0
      tech
      tv future in the hands of viewers with home th...


​    
      1
      business
      worldcom boss  left books alone  former worldc...


​    
      2
      sport
      tigers wary of farrell  gamble  leicester say ...


​    
      3
      sport
      yeading face newcastle in fa cup premiership s...


​    
      4
      entertainment
      ocean s twelve raids box office ocean s twelve...


​    
      ...
      ...
      ...


​    
      2220
      business
      cars pull down us retail figures us retail sal...


​    
      2221
      politics
      kilroy unveils immigration policy ex-chatshow ...


​    
      2222
      entertainment
      rem announce new glasgow concert us band rem h...


​    
      2223
      politics
      how political squabbles snowball it s become c...


​    
      2224
      sport
      souness delight at euro progress boss graeme s...

  


2225 rows × 2 columns​

</div>

## 토크나이저 생성



```python
from torchtext.data.utils import get_tokenizer
```

tokenizer의 타입으로는 `basic_english`, `spacy`, `moses`, `toktok`, `revtok`, `subword` 이 있습니다.



다만, 이 중 몇개의 타입은 추가 패키지가 설치되어야 정상 동작합니다.



```python
tokenizer = get_tokenizer('basic_english', language='en')
tokenizer("I'd like to learn torchtext")
```


				HTML


​					
​				
​				
​						
​				
​			
		['i', "'", 'd', 'like', 'to', 'learn', 'torchtext']


토큰 타입을 지정하면 그에 맞는 tokenizer를 반환하는 함수를 생성한 뒤 원하는 타입을 지정하여 tokenizer를 생성할 수 있습니다.



```python
def generate_tokenizer(tokenizer_type, language='en'):
    return get_tokenizer(tokenizer_type, language=language)
```

`basic_english`를 적용한 경우



```python
tokenizer = generate_tokenizer('basic_english')
tokenizer("I'd like to learn torchtext")
```


				HTML


​					
​				
​				
​						
​				
​			
		['i', "'", 'd', 'like', 'to', 'learn', 'torchtext']


`toktok`을 적용한 경우



```python
tokenizer = generate_tokenizer('toktok')
tokenizer("I'd like to learn torchtext")
```


				HTML


​					
​				
​				
​						
​				
​			
		['I', "'", 'd', 'like', 'to', 'learn', 'torchtext']


```python
from nltk.tokenize import word_tokenize

word_tokenize("I'd like to learn torchtext")
```


				HTML


​					
​				
​				
​						
​				
​			
		['I', "'d", 'like', 'to', 'learn', 'torchtext']


## 필드(Field) 정의



```python
from torchtext.legacy import data
```

`torchtext.legacy.data.Field` 

- `Field` 클래스는 `Tensor`로 변환하기 위한 지침과 함께 데이터 유형을 정의합니다. 
- `Field` 객체는 `vocab` 개체를 보유합니다.
- `Field` 객체는 토큰화 방법, 생성할 Tensor 종류와 같이 데이터 유형을 수치화하는 역할을 수행합니다.



```python
TEXT = data.Field(sequential=True,    # 순서를 반영
                  tokenize=tokenizer, # tokenizer 지정
                  fix_length=120,     # 한 문장의 최대 길이 지정
                  lower=True,         # 소문자 화
                  batch_first=True)   # batch 를 가장 먼저 출력


LABEL = data.Field(sequential=False)
```

`fields` 변수에 dictionary를 생성합니다.

- `key`: 읽어 들여올 파일의 열 이름을 지정합니다.
- `value`: (`문자열`, `data.Field`) 형식으로 지정합니다. 여기서 지정한 문자열이 나중에 생성된 data의 변수 이름으로 생성됩니다.



(참고) fields에 `[('text', TEXT), ('label', LABEL)]` 와 같이 생성하는 경우도 있습니다. 컬러명 변경이 필요하지 않은 경우는 `List(tuple(컬럼명, 변수))`로 생성할 수 있습니다.



```python
fields = {
    'text': ('text', TEXT), 
    'category': ('label', LABEL)
}
```

## 데이터셋 로드 및 분할

`TabularDataset` 클래스는 정형 데이터파일로부터 직접 데이터를 읽을 때 유용합니다.



지원하는 파일 형식은 `CSV`, `JSON`, `TSV` 을 지원합니다.



```python
import random
from torchtext.legacy.data import TabularDataset

SEED = 123

dataset = TabularDataset(path='bbc-text.csv',  # 파일의 경로
                         format='CSV',         # 형식 지정
                         fields=fields,        # 이전에 생성한 field 지정
#                          skip_header=True    # 첫 번째 행은 컬러명이므로 skip
                        )        
```

이전에 생성한 `dataset` 변수로 train / test 데이터셋을 분할 합니다.



```python
train_data, test_data = dataset.split(split_ratio=0.8,               # 분할 비율
                                      stratified=True,               # stratify 여부
                                      strata_field='label',          # stratify 대상 컬럼명
                                      random_state=random.seed(SEED) # 시드
                                     )
```

```python
# 생성된 train / test 데이터셋의 크기를 출력 합니다.
len(train_data), len(test_data)
```


				HTML


​					
​				
​				
​						
​				
​			
		(1781, 444)


## 단어 사전 생성



```python
TEXT.build_vocab(train_data, 
                 max_size=1000,             # 최대 vocab_size 지정 (미지정시 전체 단어사전 개수 대입)
                 min_freq=5,                # 최소 빈도 단어수 지정
                 vectors='glove.6B.100d')   # 워드임베딩 vector 지정, None으로 지정시 vector 사용 안함

LABEL.build_vocab(train_data)
```

```python
NUM_VOCABS = len(TEXT.vocab.stoi)
NUM_VOCABS
```


				HTML


​					
​				
​				
​						
​				
​			
		1002


```python
TEXT.vocab.freqs.most_common(10)
```


				HTML


​					
​				
​				
​						
​				
​			
		[('the', 41674),
 ('to', 19644),
 ('of', 15674),
 ('and', 14621),
 ('a', 14327),
 ('in', 13995),
 ('s', 7126),
 ('for', 7054),
 ('is', 6535),
 ('that', 6329)]
​

`TEXT.vocab.stoi`는 문자열을 index로, `TEXT.vocab.itos`는 index를 문자열로 변환합니다.



```python
TEXT.vocab.stoi
```


				HTML


​					
​				
​				
​						
​				
​			
		defaultdict(>,
	        {'': 0,
	         '': 1,
	         'the': 2,
	         'to': 3,
	         'of': 4,
	         'and': 5,
	         'a': 6,
	         'in': 7,
	         's': 8,
	         'for': 9,
	         'is': 10,
	         'that': 11,
	         'it': 12,
	         'on': 13,
	         'was': 14,
	         'said': 15,
	         'he': 16,
	         'be': 17,
	         'with': 18,
	         'has': 19,
	         'as': 20,
	         'have': 21,
	         'at': 22,
	         'by': 23,
	         'are': 24,
	         'but': 25,
	         'will': 26,
	         '.': 27,
	         'i': 28,
	         'from': 29,
	         'not': 30,
	         '-': 31,
	         'they': 32,
	         'his': 33,
	         'we': 34,
	         'mr': 35,
	         'an': 36,
	         'this': 37,
	         'had': 38,
	         'which': 39,
	         'been': 40,
	         'would': 41,
	         'their': 42,
	         'more': 43,
	         'its': 44,
	         'were': 45,
	         'also': 46,
	         ')': 47,
	         '(': 48,
	         'who': 49,
	         '%': 50,
	         'new': 51,
	         'people': 52,
	         'up': 53,
	         'us': 54,
	         'there': 55,
	         'about': 56,
	         ':': 57,
	         'one': 58,
	         'after': 59,
	         'than': 60,
	         'can': 61,
	         'or': 62,
	         'out': 63,
	         'could': 64,
	         'if': 65,
	         'year': 66,
	         'you': 67,
	         'all': 68,
	         'over': 69,
	         'said.': 70,
	         'when': 71,
	         'last': 72,
	         '£': 73,
	         'first': 74,
	         't': 75,
	         'two': 76,
	         'other': 77,
	         'into': 78,
	         '$': 79,
	         'some': 80,
	         'world': 81,
	         'government': 82,
	         'what': 83,
	         'now': 84,
	         'time': 85,
	         'she': 86,
	         'uk': 87,
	         'so': 88,
	         'against': 89,
	         'only': 90,
	         'told': 91,
	         'just': 92,
	         'being': 93,
	         'make': 94,
	         'do': 95,
	         'no': 96,
	         'get': 97,
	         'best': 98,
	         'such': 99,
	         'very': 100,
	         'many': 101,
	         'made': 102,
	         'because': 103,
	         'while': 104,
	         'labour': 105,
	         'them': 106,
	         'before': 107,
	         'like': 108,
	         'should': 109,
	         'film': 110,
	         'next': 111,
	         'years': 112,
	         'her': 113,
	         '000': 114,
	         'number': 115,
	         'game': 116,
	         'three': 117,
	         'bbc': 118,
	         'most': 119,
	         'take': 120,
	         'back': 121,
	         'any': 122,
	         'way': 123,
	         'set': 124,
	         'music': 125,
	         'our': 126,
	         'may': 127,
	         'company': 128,
	         'since': 129,
	         'my': 130,
	         'home': 131,
	         'england': 132,
	         'still': 133,
	         'well': 134,
	         'then': 135,
	         'those': 136,
	         'good': 137,
	         'how': 138,
	         'blair': 139,
	         'going': 140,
	         'million': 141,
	         'much': 142,
	         'won': 143,
	         'market': 144,
	         'down': 145,
	         'firm': 146,
	         'between': 147,
	         'think': 148,
	         'go': 149,
	         'second': 150,
	         'win': 151,
	         'work': 152,
	         'says': 153,
	         'games': 154,
	         'want': 155,
	         'did': 156,
	         'off': 157,
	         'play': 158,
	         'used': 159,
	         'part': 160,
	         'use': 161,
	         'him': 162,
	         'minister': 163,
	         'mobile': 164,
	         'both': 165,
	         'added': 166,
	         'say': 167,
	         'party': 168,
	         'top': 169,
	         'these': 170,
	         'public': 171,
	         'through': 172,
	         'european': 173,
	         'british': 174,
	         'see': 175,
	         'under': 176,
	         'show': 177,
	         'technology': 178,
	         'election': 179,
	         'even': 180,
	         'however': 181,
	         '2004': 182,
	         'sales': 183,
	         'tv': 184,
	         'where': 185,
	         'put': 186,
	         ';': 187,
	         'expected': 188,
	         'already': 189,
	         'brown': 190,
	         'four': 191,
	         'chief': 192,
	         'news': 193,
	         'end': 194,
	         'six': 195,
	         'players': 196,
	         'week': 197,
	         'during': 198,
	         'former': 199,
	         'country': 200,
	         'five': 201,
	         'growth': 202,
	         'come': 203,
	         'group': 204,
	         'according': 205,
	         'london': 206,
	         'own': 207,
	         'companies': 208,
	         'director': 209,
	         'don': 210,
	         'm': 211,
	         'plans': 212,
	         'britain': 213,
	         '10': 214,
	         'service': 215,
	         'year.': 216,
	         'help': 217,
	         'big': 218,
	         'economic': 219,
	         'including': 220,
	         'economy': 221,
	         'need': 222,
	         'bank': 223,
	         'deal': 224,
	         'digital': 225,
	         'phone': 226,
	         'international': 227,
	         'around': 228,
	         'industry': 229,
	         'another': 230,
	         'users': 231,
	         'me': 232,
	         'money': 233,
	         'too': 234,
	         'firms': 235,
	         'net': 236,
	         'record': 237,
	         'tax': 238,
	         'right': 239,
	         'came': 240,
	         'took': 241,
	         'got': 242,
	         'system': 243,
	         'services': 244,
	         'france': 245,
	         'howard': 246,
	         'know': 247,
	         'general': 248,
	         'months': 249,
	         'move': 250,
	         'place': 251,
	         'spokesman': 252,
	         'third': 253,
	         'really': 254,
	         'great': 255,
	         'open': 256,
	         'start': 257,
	         'wales': 258,
	         'report': 259,
	         'same': 260,
	         'hit': 261,
	         'business': 262,
	         'using': 263,
	         'europe': 264,
	         'president': 265,
	         'prime': 266,
	         'united': 267,
	         'despite': 268,
	         'team': 269,
	         'video': 270,
	         'foreign': 271,
	         'day': 272,
	         'ireland': 273,
	         'software': 274,
	         'o': 275,
	         'seen': 276,
	         'your': 277,
	         'china': 278,
	         'without': 279,
	         'better': 280,
	         'court': 281,
	         'likely': 282,
	         'national': 283,
	         'given': 284,
	         'club': 285,
	         'give': 286,
	         'lost': 287,
	         'oil': 288,
	         'saying': 289,
	         'far': 290,
	         'media': 291,
	         'office': 292,
	         'called': 293,
	         'able': 294,
	         'figures': 295,
	         'final': 296,
	         'found': 297,
	         'side': 298,
	         'data': 299,
	         'every': 300,
	         'real': 301,
	         'security': 302,
	         've': 303,
	         'although': 304,
	         'whether': 305,
	         'high': 306,
	         'legal': 307,
	         '2005': 308,
	         'long': 309,
	         'state': 310,
	         'decision': 311,
	         'become': 312,
	         'cup': 313,
	         'lot': 314,
	         'offer': 315,
	         'information': 316,
	         'life': 317,
	         'michael': 318,
	         'recent': 319,
	         'computer': 320,
	         'executive': 321,
	         'online': 322,
	         'campaign': 323,
	         'internet': 324,
	         'lord': 325,
	         'radio': 326,
	         'due': 327,
	         'early': 328,
	         'taking': 329,
	         'making': 330,
	         'biggest': 331,
	         'law': 332,
	         'does': 333,
	         'played': 334,
	         'action': 335,
	         'away': 336,
	         'later': 337,
	         'star': 338,
	         'went': 339,
	         'look': 340,
	         'looking': 341,
	         'future': 342,
	         'countries': 343,
	         'few': 344,
	         'less': 345,
	         'hard': 346,
	         'playing': 347,
	         'currently': 348,
	         'eu': 349,
	         'major': 350,
	         'rise': 351,
	         'secretary': 352,
	         'times': 353,
	         'current': 354,
	         'month': 355,
	         'analysts': 356,
	         'cut': 357,
	         'increase': 358,
	         'john': 359,
	         'player': 360,
	         'shares': 361,
	         'past': 362,
	         '2003': 363,
	         'little': 364,
	         'tory': 365,
	         'david': 366,
	         'held': 367,
	         'interest': 368,
	         'role': 369,
	         'financial': 370,
	         'house': 371,
	         'leader': 372,
	         'prices': 373,
	         'following': 374,
	         'chancellor': 375,
	         'awards': 376,
	         'case': 377,
	         'different': 378,
	         'latest': 379,
	         'december': 380,
	         'face': 381,
	         'left': 382,
	         'local': 383,
	         'south': 384,
	         'spending': 385,
	         'broadband': 386,
	         'having': 387,
	         'strong': 388,
	         'until': 389,
	         'among': 390,
	         'each': 391,
	         'it.': 392,
	         'pay': 393,
	         'support': 394,
	         '2': 395,
	         'children': 396,
	         'further': 397,
	         're': 398,
	         'thought': 399,
	         'earlier': 400,
	         'important': 401,
	         '20': 402,
	         'nations': 403,
	         'never': 404,
	         'research': 405,
	         'working': 406,
	         'almost': 407,
	         'meeting': 408,
	         'rights': 409,
	         'band': 410,
	         'believe': 411,
	         'bill': 412,
	         'find': 413,
	         'microsoft': 414,
	         'run': 415,
	         'personal': 416,
	         'rate': 417,
	         'ahead': 418,
	         'match': 419,
	         'key': 420,
	         'must': 421,
	         'return': 422,
	         'again': 423,
	         'include': 424,
	         'line': 425,
	         'ms': 426,
	         'series': 427,
	         'days': 428,
	         'something': 429,
	         'always': 430,
	         'award': 431,
	         'cost': 432,
	         'keep': 433,
	         'trade': 434,
	         'consumer': 435,
	         'am': 436,
	         'sold': 437,
	         'try': 438,
	         'control': 439,
	         'half': 440,
	         'announced': 441,
	         'change': 442,
	         'january': 443,
	         'rather': 444,
	         '12': 445,
	         'scotland': 446,
	         'things': 447,
	         'behind': 448,
	         'japan': 449,
	         'manager': 450,
	         'tony': 451,
	         'women': 452,
	         'added.': 453,
	         'films': 454,
	         'across': 455,
	         'lead': 456,
	         'police': 457,
	         'access': 458,
	         'man': 459,
	         'chance': 460,
	         'claims': 461,
	         'once': 462,
	         'search': 463,
	         'yet': 464,
	         'means': 465,
	         'issue': 466,
	         'power': 467,
	         'taken': 468,
	         'sale': 469,
	         'allow': 470,
	         'coach': 471,
	         'huge': 472,
	         'american': 473,
	         'global': 474,
	         '1': 475,
	         'actor': 476,
	         'council': 477,
	         'dollar': 478,
	         'getting': 479,
	         'india': 480,
	         'men': 481,
	         'released': 482,
	         'rugby': 483,
	         'sunday': 484,
	         'tories': 485,
	         'share': 486,
	         'beat': 487,
	         'job': 488,
	         'jobs': 489,
	         'plan': 490,
	         'within': 491,
	         'available': 492,
	         'might': 493,
	         'problems': 494,
	         'free': 495,
	         'full': 496,
	         'mark': 497,
	         'content': 498,
	         'enough': 499,
	         '100': 500,
	         'done': 501,
	         'members': 502,
	         'point': 503,
	         'trying': 504,
	         'coming': 505,
	         'human': 506,
	         'result': 507,
	         'young': 508,
	         'investment': 509,
	         'least': 510,
	         'political': 511,
	         'clear': 512,
	         'phones': 513,
	         'problem': 514,
	         'saw': 515,
	         'budget': 516,
	         'chairman': 517,
	         'continue': 518,
	         'french': 519,
	         '5': 520,
	         'doing': 521,
	         'total': 522,
	         'years.': 523,
	         'ever': 524,
	         'price': 525,
	         'winning': 526,
	         'fans': 527,
	         'possible': 528,
	         'several': 529,
	         'sony': 530,
	         'demand': 531,
	         'here': 532,
	         'needed': 533,
	         'showed': 534,
	         'title': 535,
	         'issues': 536,
	         'main': 537,
	         'minutes': 538,
	         'wanted': 539,
	         'close': 540,
	         'based': 541,
	         'chelsea': 542,
	         'war': 543,
	         'apple': 544,
	         '50': 545,
	         'centre': 546,
	         'development': 547,
	         'hold': 548,
	         'ministers': 549,
	         'performance': 550,
	         'stock': 551,
	         'warned': 552,
	         'march': 553,
	         'pc': 554,
	         'williams': 555,
	         'annual': 556,
	         'asked': 557,
	         'commission': 558,
	         'd': 559,
	         'lib': 560,
	         'mps': 561,
	         'previous': 562,
	         'quarter': 563,
	         'sir': 564,
	         'though': 565,
	         'weeks': 566,
	         '3': 567,
	         'meet': 568,
	         'union': 569,
	         'bid': 570,
	         'forward': 571,
	         'head': 572,
	         'injury': 573,
	         'let': 574,
	         'recently': 575,
	         'small': 576,
	         'statement': 577,
	         '4': 578,
	         'agreed': 579,
	         'consumers': 580,
	         'costs': 581,
	         'book': 582,
	         'fact': 583,
	         'led': 584,
	         '30': 585,
	         'committee': 586,
	         'level': 587,
	         'network': 588,
	         'production': 589,
	         'reports': 590,
	         'failed': 591,
	         'victory': 592,
	         'why': 593,
	         'buy': 594,
	         'city': 595,
	         'liberal': 596,
	         'comes': 597,
	         'league': 598,
	         'rates': 599,
	         'season': 600,
	         'wants': 601,
	         'cash': 602,
	         'iraq': 603,
	         'tuesday': 604,
	         'cannot': 605,
	         'claim': 606,
	         'evidence': 607,
	         'policy': 608,
	         'eight': 609,
	         'programme': 610,
	         'rose': 611,
	         'vote': 612,
	         '11': 613,
	         'act': 614,
	         'charles': 615,
	         'difficult': 616,
	         'form': 617,
	         'giant': 618,
	         'november': 619,
	         'late': 620,
	         'running': 621,
	         'russian': 622,
	         'single': 623,
	         'thing': 624,
	         'ago': 625,
	         'health': 626,
	         'hope': 627,
	         'italy': 628,
	         'sure': 629,
	         'yukos': 630,
	         'success': 631,
	         'wednesday': 632,
	         'aid': 633,
	         'arsenal': 634,
	         'call': 635,
	         'friday': 636,
	         'list': 637,
	         'paul': 638,
	         'received': 639,
	         'rules': 640,
	         'competition': 641,
	         'fell': 642,
	         'jones': 643,
	         'low': 644,
	         'talks': 645,
	         'champion': 646,
	         'didn': 647,
	         'gaming': 648,
	         'growing': 649,
	         'networks': 650,
	         'per': 651,
	         'race': 652,
	         'singer': 653,
	         'today': 654,
	         'tour': 655,
	         'boost': 656,
	         'february': 657,
	         'nothing': 658,
	         'project': 659,
	         'television': 660,
	         '15': 661,
	         'australian': 662,
	         'choice': 663,
	         'christmas': 664,
	         'gordon': 665,
	         'higher': 666,
	         'launch': 667,
	         'launched': 668,
	         'mean': 669,
	         '...': 670,
	         'denied': 671,
	         'version': 672,
	         'web': 673,
	         '2004.': 674,
	         'fall': 675,
	         'time.': 676,
	         'v': 677,
	         'boss': 678,
	         'ensure': 679,
	         'sent': 680,
	         '2001': 681,
	         '2003.': 682,
	         'actress': 683,
	         'calls': 684,
	         'career': 685,
	         'debt': 686,
	         'fourth': 687,
	         'known': 688,
	         'leading': 689,
	         'profits': 690,
	         'quite': 691,
	         'reported': 692,
	         'stop': 693,
	         'album': 694,
	         'ball': 695,
	         'instead': 696,
	         'martin': 697,
	         'opening': 698,
	         'remain': 699,
	         'started': 700,
	         'dr': 701,
	         'involved': 702,
	         'monday': 703,
	         'points': 704,
	         'popular': 705,
	         'similar': 706,
	         'sites': 707,
	         'special': 708,
	         'africa': 709,
	         'customers': 710,
	         'family': 711,
	         'large': 712,
	         'makes': 713,
	         'site': 714,
	         '2005.': 715,
	         'accused': 716,
	         'age': 717,
	         'attacks': 718,
	         'devices': 719,
	         'dvd': 720,
	         'helped': 721,
	         'provide': 722,
	         'senior': 723,
	         'board': 724,
	         'education': 725,
	         'week.': 726,
	         'event': 727,
	         'feel': 728,
	         'grand': 729,
	         'irish': 730,
	         'official': 731,
	         'olympic': 732,
	         'order': 733,
	         'bit': 734,
	         'euros': 735,
	         'live': 736,
	         'others': 737,
	         'saturday': 738,
	         'simply': 739,
	         'trial': 740,
	         'turn': 741,
	         'website': 742,
	         'break': 743,
	         'changes': 744,
	         'compared': 745,
	         'pressure': 746,
	         'robinson': 747,
	         'school': 748,
	         'short': 749,
	         'them.': 750,
	         'voters': 751,
	         'movie': 752,
	         'happy': 753,
	         'manchester': 754,
	         'shows': 755,
	         'together': 756,
	         '18': 757,
	         'fight': 758,
	         'independent': 759,
	         'paid': 760,
	         'position': 761,
	         'value': 762,
	         'via': 763,
	         'york': 764,
	         'appeal': 765,
	         'association': 766,
	         'became': 767,
	         'gave': 768,
	         'idea': 769,
	         'scottish': 770,
	         'survey': 771,
	         'themselves': 772,
	         'box': 773,
	         'football': 774,
	         'needs': 775,
	         'september': 776,
	         'thursday': 777,
	         'whole': 778,
	         'focus': 779,
	         'meanwhile': 780,
	         'named': 781,
	         'sport': 782,
	         'average': 783,
	         'date': 784,
	         'e-mail': 785,
	         'forced': 786,
	         'included': 787,
	         'private': 788,
	         'release': 789,
	         'roddick': 790,
	         'claimed': 791,
	         'comedy': 792,
	         'conference': 793,
	         'deutsche': 794,
	         'download': 795,
	         'russia': 796,
	         'seven': 797,
	         'worked': 798,
	         'bad': 799,
	         'bought': 800,
	         'create': 801,
	         'decided': 802,
	         'german': 803,
	         'liverpool': 804,
	         'october': 805,
	         'prize': 806,
	         'rock': 807,
	         'speech': 808,
	         'street': 809,
	         '25': 810,
	         'believes': 811,
	         'immigration': 812,
	         'results': 813,
	         'stars': 814,
	         'takes': 815,
	         'turned': 816,
	         'west': 817,
	         '40': 818,
	         'america': 819,
	         'conservative': 820,
	         'entertainment': 821,
	         'everything': 822,
	         'products': 823,
	         'proposals': 824,
	         'raise': 825,
	         'view': 826,
	         'exchange': 827,
	         'hopes': 828,
	         'kennedy': 829,
	         'love': 830,
	         'old': 831,
	         'potential': 832,
	         'andy': 833,
	         'battle': 834,
	         'bring': 835,
	         'car': 836,
	         'force': 837,
	         'game.': 838,
	         'germany': 839,
	         'remains': 840,
	         'test': 841,
	         'trust': 842,
	         'areas': 843,
	         'either': 844,
	         'extra': 845,
	         'goal': 846,
	         'investors': 847,
	         'often': 848,
	         'original': 849,
	         'concerned': 850,
	         'everyone': 851,
	         'll': 852,
	         'lords': 853,
	         'numbers': 854,
	         'particularly': 855,
	         'parties': 856,
	         'soon': 857,
	         'stage': 858,
	         'tough': 859,
	         'festival': 860,
	         'groups': 861,
	         'largest': 862,
	         'north': 863,
	         'poor': 864,
	         'squad': 865,
	         'amount': 866,
	         'card': 867,
	         'comments': 868,
	         'confirmed': 869,
	         'davis': 870,
	         'officials': 871,
	         'parliament': 872,
	         '2000': 873,
	         'alan': 874,
	         'australia': 875,
	         'brought': 876,
	         'charge': 877,
	         'commons': 878,
	         'concerns': 879,
	         'hollywood': 880,
	         'looked': 881,
	         'name': 882,
	         'press': 883,
	         'revealed': 884,
	         'whose': 885,
	         'admitted': 886,
	         'black': 887,
	         'cards': 888,
	         'civil': 889,
	         'course': 890,
	         'double': 891,
	         'election.': 892,
	         'energy': 893,
	         'host': 894,
	         'hours': 895,
	         'impact': 896,
	         'previously': 897,
	         'response': 898,
	         'shown': 899,
	         'smith': 900,
	         'spain': 901,
	         'stay': 902,
	         'us.': 903,
	         'workers': 904,
	         'bush': 905,
	         'care': 906,
	         'increased': 907,
	         'stand': 908,
	         'standard': 909,
	         'target': 910,
	         'widely': 911,
	         'insisted': 912,
	         'newspaper': 913,
	         'night': 914,
	         'oscar': 915,
	         'serious': 916,
	         'spend': 917,
	         'university': 918,
	         'anything': 919,
	         'confidence': 920,
	         'department': 921,
	         'drugs': 922,
	         'front': 923,
	         'fund': 924,
	         'member': 925,
	         'mp': 926,
	         'offered': 927,
	         'outside': 928,
	         'period': 929,
	         'reached': 930,
	         'spent': 931,
	         'windows': 932,
	         '14': 933,
	         '2002': 934,
	         'ban': 935,
	         'community': 936,
	         'credit': 937,
	         'critics': 938,
	         'finance': 939,
	         'japanese': 940,
	         'looks': 941,
	         'lower': 942,
	         'process': 943,
	         'signed': 944,
	         'social': 945,
	         'step': 946,
	         'ask': 947,
	         'asylum': 948,
	         'attack': 949,
	         'banks': 950,
	         'drive': 951,
	         'felt': 952,
	         'gone': 953,
	         'indian': 954,
	         'majority': 955,
	         'markets': 956,
	         'message': 957,
	         'mike': 958,
	         'moment': 959,
	         'risk': 960,
	         'virus': 961,
	         '6': 962,
	         'agency': 963,
	         'air': 964,
	         'analyst': 965,
	         'authorities': 966,
	         'believed': 967,
	         'body': 968,
	         'central': 969,
	         'domestic': 970,
	         'google': 971,
	         'itself': 972,
	         'james': 973,
	         'nintendo': 974,
	         'reach': 975,
	         'robert': 976,
	         'sector': 977,
	         'song': 978,
	         'speaking': 979,
	         'summer': 980,
	         'taxes': 981,
	         'technologies': 982,
	         'began': 983,
	         'bt': 984,
	         'chart': 985,
	         'defence': 986,
	         'details': 987,
	         'followed': 988,
	         'opportunity': 989,
	         'shot': 990,
	         'suggested': 991,
	         '&': 992,
	         '16': 993,
	         'along': 994,
	         'dems': 995,
	         'laws': 996,
	         'rival': 997,
	         'story': 998,
	         'watch': 999,
	         ...})


```python
# string to index
print(TEXT.vocab.stoi['this'])
print(TEXT.vocab.stoi['pretty'])
print(TEXT.vocab.stoi['original'])

print('==='*10)

# index to string
print(TEXT.vocab.itos[14])
print(TEXT.vocab.itos[194])
print(TEXT.vocab.itos[237])
```


				HTML


​					
​				
​				
​						
​				
​			
		37
0
849
==============================
was
end
record
​

## 버킷 이터레이터 생성



- `BucketIterator` 의 주된 역할은 데이터셋에 대한 배치 구성입니다.



```python
import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),     # dataset
    sort=False,
    repeat=False,
    batch_size=BATCH_SIZE,       # 배치사이즈
    device=device)               # device 지정
```

1개의 배치를 추출합니다.



```python
# 1개의 batch 추출
sample_data = next(iter(train_iterator))
```

`text` 의 shape 를 확인합니다.



```python
# batch_size, sequence_length
sample_data.text.shape
```


				HTML


​					
​				
​				
​						
​				
​			
		torch.Size([32, 120])


```python
len(sample_data.text)
```


				HTML


​					
​				
​				
​						
​				
​			
		32


```python
sample_data.label.size(0)
```


				HTML


​					
​				
​				
​						
​				
​			
		32


`label` 의 shape 를 확인합니다.



```python
# batch_size
sample_data.label.shape
```


				HTML


​					
​				
​				
​						
​				
​			
		torch.Size([32])


```python
# label을 출력합니다.
sample_data.label
```


				HTML


​					
​				
​				
​						
​				
​			
		tensor([5, 1, 2, 4, 1, 4, 5, 2, 2, 4, 1, 2, 5, 3, 1, 3, 4, 4, 1, 4, 3, 3, 2, 1,
	    3, 5, 2, 4, 1, 5, 3, 5], device='cuda:1')


아래에서 확인할 수 있듯이 `<unk>` 토큰 때문에 카테고리의 개수가 5개임에도 불구하고 index는 0번부터 5번까지 맵핑되어 있습니다.



```python
LABEL.vocab.stoi
```


				HTML


​					
​				
​				
​						
​				
​			
		defaultdict(>,
	        {'': 0,
	         'sport': 1,
	         'business': 2,
	         'politics': 3,
	         'tech': 4,
	         'entertainment': 5})


따라서, 0번을 무시해주기 위해서는 배치 학습시 다음과 같이 처리해 줄 수 있습니다.



1을 subtract 해줌으로써 0~4번 index로 조정해 주는 것입니다.



```python
sample_data.label.sub_(1)
```


				HTML


​					
​				
​				
​						
​				
​			
		tensor([4, 0, 1, 3, 0, 3, 4, 1, 1, 3, 0, 1, 4, 2, 0, 2, 3, 3, 0, 3, 2, 2, 1, 0,
	    2, 4, 1, 3, 0, 4, 2, 4], device='cuda:1')


## 데이터프레임(DataFrame) 커스텀 데이터셋 클래스



`torchtext.legacy.data.Dataset`을 확장하여 DataFrame을 바로 `BucketIterator`로 변환할 수 있습니다.



```python
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 123

# 데이터프레임을 로드 합니다.
df = pd.read_csv('bbc-text.csv')

# 컬럼명은 text / label 로 변경합니다
df = df.rename(columns={'category': 'label'})
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
      label
      text

  


​    
      0
      tech
      tv future in the hands of viewers with home th...


​    
      1
      business
      worldcom boss  left books alone  former worldc...


​    
      2
      sport
      tigers wary of farrell  gamble  leicester say ...


​    
      3
      sport
      yeading face newcastle in fa cup premiership s...


​    
      4
      entertainment
      ocean s twelve raids box office ocean s twelve...


​    
      ...
      ...
      ...


​    
      2220
      business
      cars pull down us retail figures us retail sal...


​    
      2221
      politics
      kilroy unveils immigration policy ex-chatshow ...


​    
      2222
      entertainment
      rem announce new glasgow concert us band rem h...


​    
      2223
      politics
      how political squabbles snowball it s become c...


​    
      2224
      sport
      souness delight at euro progress boss graeme s...

  


2225 rows × 2 columns​

</div>



```python
# train / validation 을 분할 합니다.
train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
```

```python
# train DataFrame
train_df.head()
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
      label
      text

  


​    
      1983
      sport
      officials respond in court row australian tenn...


​    
      878
      tech
      slow start to speedy net services faster broad...


​    
      94
      politics
      amnesty chief laments war failure the lack of ...


​    
      1808
      sport
      dal maso in to replace bergamasco david dal ma...


​    
      1742
      tech
      technology gets the creative bug the hi-tech a...

  



</div>



```python
# validation DataFrame
val_df.head()
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
      label
      text

  


​    
      717
      politics
      child access laws shake-up parents who refuse ...


​    
      798
      entertainment
      fry set for role in hitchhiker s actor stephen...


​    
      1330
      business
      palestinian economy in decline despite a short...


​    
      18
      business
      japanese banking battle at an end japan s sumi...


​    
      1391
      business
      manufacturing recovery  slowing  uk manufactur...

  



</div>



```python
# 필요한 모듈 import
import torch
from torchtext.legacy import data
from torchtext.data.utils import get_tokenizer

# device 설정
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
		cuda:1


`torchtext.legacy.data.Dataset`을 상속하여 데이터프레임을 로드할 수 있습니다.



```python
class DataFrameDataset(data.Dataset):

    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            # text, label 컬럼명은 필요시 변경하여 사용합니다
            label = row['label'] if not is_test else None
            text = row['text'] 
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, False, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)
```

```python
# 토크나이저 정의 (다른 토크나이저로 대체 가능)
tokenizer = get_tokenizer('basic_english')
```

앞선 내용과 마찬가지로 `Field`를 구성합니다.



```python
TEXT = data.Field(sequential=True,    # 순서를 반영
                  tokenize=tokenizer, # tokenizer 지정
                  fix_length=120,     # 한 문장의 최대 길이 지정
                  lower=True,         # 소문자화
                  batch_first=True)   # batch 를 가장 먼저 출력


LABEL = data.Field(sequential=False)

# fiels 변수에 List(tuple(컬럼명, 변수)) 형식으로 구성 후 대입
fields = [('text', TEXT), ('label', LABEL)]
```

```python
# DataFrame의 Splits로 데이터셋 분할
train_ds, val_ds = DataFrameDataset.splits(fields, train_df=train_df, val_df=val_df)
```

```python
# 단어 사전 생성
TEXT.build_vocab(train_ds, 
                 max_size=1000,             # 최대 vocab_size 지정 (미지정시 전체 단어사전 개수 대입)
                 min_freq=5,                # 최소 빈도 단어수 지정
                 vectors='glove.6B.100d')   # 워드임베딩 vector 지정, None으로 지정시 vector 사용 안함

LABEL.build_vocab(train_ds)
```

```python
# 단어 사전 개수 출력
NUM_VOCABS = len(TEXT.vocab)
NUM_VOCABS
# 개수 1000 + <unk> + <pad> : 총 1002개
```


				HTML


​					
​				
​				
​						
​				
​			
		1002


`BucketIterator`를 생성합니다.



```python
BATCH_SIZE = 32

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_ds, val_ds), 
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=device)
```

```python
# 1개 배치 추출
sample_data = next(iter(train_iterator))
```

```python
# text shape 출력 (batch_size, sequence_length)
sample_data.text.shape
```


				HTML


​					
​				
​				
​						
​				
​			
		torch.Size([32, 120])


```python
# label 출력 (batch)
sample_data.label
```


				HTML


​					
​				
​				
​						
​				
​			
		tensor([1, 2, 4, 4, 3, 4, 5, 4, 5, 1, 2, 1, 2, 2, 5, 5, 2, 5, 5, 2, 5, 1, 1, 2,
	    5, 5, 1, 3, 2, 3, 3, 5], device='cuda:1')
