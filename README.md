


# Large and small-scale models’ fusion-driven proactive robotic manipulation control for human-robot collaborative assembly in Industry 5.0

## Introduction 

This is our work on combining large and small models to achieve robot control, thanks to  “ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation” for inspiring our work.

## Usage

1. Clone this repository with: 
```git clone https://github.com/mdx-box/LLM_for_robot.git```

2. Download Isaac Sim and config it properly. 

3. Create conda environment
```
conda create -n LSM_SLM python=3.11
conda activate LSM_SLM
pip install -r requirements.txt

```

4. Config the config files according your demands.

5. Start the project. Here, we provide two choice, namely automated one and human-assisted one.

```
auto: python main_test.py
human: python main_human.py

```
6. Notes: How to define your scene to Isaac Sim？

6.1. Create your scene with modelling software, such as Solidworks, FreeCAD, Blender, etc.

6.2 Expert your model with .gltf format. And then import this model to Blender. Then set geomerty to origin. And then export it with the .usd format.

![Model setting in Blender](https://github.com/mdx-box/LLM_for_robot/blob/main/Figure/Blender.jpg)

6.2 Open Isaac Sim platform. Open exported .usd file. Currently, there is only a model without any physical characteristic. Therefore, we we should add more details. Firstly, it is recommended to add your desired color to the object. 

![Usd Import](https://github.com/mdx-box/LLM_for_robot/blob/main/Figure/usd_expoert.jpg)
![Color setting](https://github.com/mdx-box/LLM_for_robot/blob/main/Figure/color.jpg)
![Color setting](https://github.com/mdx-box/LLM_for_robot/blob/main/Figure/color_set.jpg)

6.3 Copy the visual one and rename it with collusion. Move 'collusion' into the same folder as 'visual'. Here, 'visual' refers to visualization, making it visible when opened. And "collusion" is used to add physical property. 

![Same foler](https://github.com/mdx-box/LLM_for_robot/blob/main/Figure/same_folder.jpg)

6.4 Add the "rigid property" to collusion. 

![Rigid property](https://github.com/mdx-box/LLM_for_robot/blob/main/Figure/rigid.jpg)

6.5 Finally, saving your model as .usd file. And put its address in config file.

![Final model](https://github.com/mdx-box/LLM_for_robot/blob/main/Figure/Iss_final.jpg)

7. The hardware requirement of Isaac Sim is relatively high. In the future, we will use ![Genesis](https://github.com/Genesis-Embodied-AI/Genesis) to codnuct our task!


