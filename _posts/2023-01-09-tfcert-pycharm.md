---
layout: page
title: "[2023년 업데이트] 구글 텐서플로 개발자 자격인증 시험을 위한 환경설치(PyCharm, 필요 라이브러리 설치)"
description: "[2023년 업데이트] 구글 텐서플로 개발자 자격인증 시험을 위한 환경설치(PyCharm, 필요 라이브러리 설치)에 대해 알아보겠습니다."
headline: "[2023년 업데이트] 구글 텐서플로 개발자 자격인증 시험을 위한 환경설치(PyCharm, 필요 라이브러리 설치)에 대해 알아보겠습니다."
categories: TensorFlow
tags: [python, 파이썬, tensorflow, tensorflow certificate, google tensorflow, 텐서플로 자격증, 텐서플로우 자격증, 파이참 설치, 가상환경 설정, 아나콘다 가상환경, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터, 테디노트]
comments: true
published: true
typora-copy-images-to: ../images/2023-01-09
---

본 포스팅은 **Google TensorFlow Developers Certificate** 자격인증 시험을 위한 환경설치를 위한 내용입니다.



주요 내용은 다음과 같습니다.

- STEP 1: 아나콘다(Anaconda) 가상환경 설치
- STEP 2: PyCharm 설치
- STEP 3: 가상환경 생성 후 필요한 라이브러리 설치
- STEP 4: PyCharm에 설치한 가상환경 적용



STEP 2 ~ STEP 4의 내용은 설정과정을 글로 풀어내기 다소 어려운 점이 있어 유튜브 영상과 같이 참고하여 진행해 주시면 됩니다.



## STEP 1. 아나콘다 가상환경을 다운로드 후 설치를 진행합니다.

![anaconda](../images/2023-01-09/anaconda.png)

아래 링크를 클릭하여 OS에 맞는 **아나콘다 가상환경을 다운로드 받고 설치를 진행**합니다.

- **Windows**: https://www.anaconda.com/distribution/#windows
- **Mac OS**: https://www.anaconda.com/distribution/#macos

**64-Bit Graphical Installer**를 인스톨하시면 됩니다.

## 

## STEP 2 ~ STEP 4. 아래 YouTube 영상을 참고하여 진행해 주세요!



<br>

<br>

<iframe width="560" height="315" src="https://www.youtube.com/embed/Sotje18bINY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<br>

<br>

1. **가상환경 설치 패키지 목록**

   **영상을 따라서 진행**해 주시면 됩니다만, 설치하는 패키지 목록은 (영상이 아닌) **아래에 표기된 최신 버전**을 설치해 주셔야 합니다.

   **설치에 활용한 패키지 목록 (최신 업데이트 2023. 01. 08)**

   ```bash
   pip install tensorflow==2.9.0
   pip install tensorflow-datasets==4.6.0
   pip install numpy==1.22.4
   pip install Pillow==9.1.1
   pip install scipy==1.7.3
   pip install pandas==1.4.2
   pip install urllib3
   ```

   

2. [**PyCharm 설치 다운로드 링크**](https://www.jetbrains.com/ko-kr/pycharm/download/)



## 구글 텐서플로 자격 인증 시험 강의

끝으로, 제가 직접 제작한 구글 텐서플로 자격 인증 시험 대비 강의 링크를 공유합니다.

입문자 분들을 대상으로 쉽게 풀어낸 강의입니다. 이미 어느정도 텐서플로를 다룰 줄 아시는 분들은 수강하실 필요 없습니다.

강의를 수강하시는 분들은 비공개 슬랙 채널로 초대장을 발송해 드립니다. 슬랙 채널에서 시험 관련 질문을 주시면 1:1 피드백을 드리고 있습니다.

- [인프런](https://www.inflearn.com/course/%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0-%EC%9E%90%EA%B2%A9%EC%A6%9D)
- [런어데이](https://learnaday.kr/open-course/tfcert)



**감사합니다! 모두 합격 하세요!**





