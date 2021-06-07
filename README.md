![KakaoTalk_20210531_124413788](https://user-images.githubusercontent.com/79948405/120136815-f5416880-c20d-11eb-8947-3f6304fb9f17.jpg)


# YOLO  Object Detector를 활용한 화재탐지 시스템 개발

### CCTV영상처리를 통해 FIRE & SMOKE 객체 검출
가장 보편화된 AHD(Analog High Definition)카메라로 촬영된 영상에서 불과 연기를 탐지하는 것이 주요 목표입니다

### 솔루션 모색
  - CNN 기반의 VGG16, Resnet을 활용한 분류모델
  - YOLO v3, v5, Scaled v4,
  - openCV 
  
솔루션이 적용되었습니다

선정기준은 게재된 논문, 구글링, 캐글을 통해 가장 성능이 좋았던 솔루션 등이며 object detection 의 기본으로 알려진 Faster R-CNN은 성능은 훌륭하지만 Real-Time 목적에 맞지 않아 제외되었습니다




### 학습용 이미지
구글링을 통해 fire and smoke dataset 검색어로 오픈소스데이터를 사용했습니다
http://smoke.ustc.edu.cn/datasets.html

에 들어가셔서 원하시는 데이터셋을 선택한 후 별도의 annotation 작업을 진행하셔도 좋습니다

사용된 도구는 Roboflow annotator이며 간단한 회원가입과 함께 사용이 가능합니다




### 솔루션별 문제점

#### Resnet, VGG16
- Classification에 특화된 모델이라 불과 영상을 동시에 탐지 못합니다
- 불과 연기과 함께 발생 할 경우 Frame을 차지하는 크기가 큰 객체를 분류합니다
- Resnet의 모델이 VGG16에 비해 월등이 높은 성능을 보입니다
- Resnet분류기의 경우 파라미터수가 23,000,000의 수준을 보이지만 학습속도는 vgg분류기와 비슷했습니다

#### yolov3
- 작은객체 검출 문제 : feature extraction에서 발생되눈 문제로 yolov3의 고질적인 문제입니다.


### Scaled YOLOv4를 활용

### 소감
저희 팀은 오픈소스솔루션과 기본지식을 통해 시도한 탐지모델을 통해 문제를 확인하고, 
Scaled YOLOv4를 사용하며 문제를 해결할 때 객체탐지의 성능을 올리기 위해 사용하는 수많은 기법에  대해 알 수 있었으며, 
결과를 통해 나오는 영상을 통해 객체탐지모델의 어느 단계에서 문제가 발생하는지 판단할 수 있는 기법에 대해 알게되었습니다

이는 추후 프로젝트 주제및 발생 문제별 상황에 맞는 문제를 해결할 수 있는 소양을 쌓을 수 있게 되었습니다.
