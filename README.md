# AMIGA_fruit_counting
Fruit detection and tracking for yield estimation using [AMIGA robot](https://farm-ng.com/).


## Introduction
This project is a python implementation for fruit detection and tracking using [YOLOv5](https://github.com/ultralytics/yolov5) and [ByteTrack](https://github.com/ifzhang/ByteTrack). The project was build to be tested jointly with [AMIGA robot](https://farm-ng.com/) for in-field fruit conting purposes. It will be tested in the [Farm@thon](https://www.lleidadrone.com/2022/09/amiga-farmthon-fira-de-sant-miquel-2022.html) organized by [Lleida Drone](https://www.lleidadrone.com). The code was prepared to be used with a [ZED](https://github.com/stereolabs/zed-python-api) camera, but it could be easily adapted for other cameras such as [Oak-D](https://github.com/luxonis/depthai-hardware/tree/master/NG2094_OAK-D-PRO-W-DEV). 


## Preparation 

First of all, create a new project folder and clone the code inside:
```
mkdir new_project
cd new_project
git clone https://github.com/GRAP-UdL-AT/AMIGA_fruit_counting.git
```

Then, install the project requirements:
```
cd AMIGA_fruit_counting
pip install -r requirements.txt
```
Clone the YOLOv5 repository and install it following the instructions at [YOLOv5 repository](https://github.com/ultralytics/yolov5):
```
cd yolov5
git clone https://github.com/ultralytics/yolov5.git
```

### Data Preparation

Make a new folder were yolo weights will be saved:
```
cd ..
cd ..
mkdir yolo_weights
```
Then, train a YOLO model or download the following [pretrained weights](https://drive.google.com/file/d/192nAusRE4WtHu5nNrnaJxE5V61AYxt7f/view?usp=sharing).



### Launch the code

* Execute the file `/new_project/AMIGA_fruit_counting/zed_fruit_counting.py`


## Authorship

This project is contributed by Francesc Net Barnes, Marc Felip Pomés and Jordi Gené-Mola from [GRAP-UdL-AT](http://www.grap.udl.cat/en/index.html).

Please contact authors to report bugs @ jordi.genemola@udl.cat


#### Acknowledgements
This work was partly funded by the Spanish Ministry of Science, Innovation and Universities (grant RTI2018-094222-B-I00[[PAgFRUIT project]]( https://www.pagfruit.udl.cat/en/) by MCIN/AEI/10.13039/501100011033 and by “ERDF, a way of making Europe”, by the European Union).
