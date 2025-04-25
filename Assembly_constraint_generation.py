import base64
from openai import OpenAI
import os
import cv2
import json
import parse
import numpy as np
import time
from datetime import datetime

# Function to encode the image
apikey = "sk-8908f398de8e41368f1fd7cecd89c476"
#将图像转换成base64编码，因为大模型只支持这种格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class ConstraintGenerator:
    def __init__(self, config):
        self.config = config
        # self.client = OpenAI(api_key=apikey)
        self.client = OpenAI(
                base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    # sk-xxx替换为自己的key
                api_key=apikey
)
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './vlm_query')
        with open(os.path.join(self.base_dir, 'prompt_template.txt'), 'r') as f:
            self.prompt_template = f.read()

    def _build_prompt(self, image_path, instruction):
        #这段代码的主要功能是构建一个包含图像和指令的提示消息，用于后续的处理，例如发送到 OpenAI API 进行处理
        #调用 encode_image 函数将图像编码为 Base64 字符串
        img_base64 = encode_image(image_path)
        #使用 prompt_template 格式化字符串，将 instruction 插入到模板中。
        prompt_text = self.prompt_template.format(instruction=instruction)
        # save prompt
        with open(os.path.join(self.task_dir, 'prompt.txt'), 'w') as f:
            f.write(prompt_text)
        #构建一个包含用户角色和内容的消息列表。内容包括格式化后的提示文本和编码后的图像 URL。
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_template.format(instruction=instruction)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                ]
            }
        ]
        return messages

    def _parse_and_save_constraints(self, output, save_dir):
        # 该方法用于解析 OpenAI API 返回的输出，并将解析后的约束条件保存到文件中
        lines = output.split("\n")
        functions = dict()
        #如果行以 def 开头，则表示这是一个函数定义的开始，记录下函数的起始行 start 和函数名 name。
        # 如果行以 return 开头，则表示这是函数定义的结束，记录下结束行 end。然后将函数的起始行到结束行之间的所有行作为函数的代码块，
        # 保存到 functions 字典中，键为函数名。
        for i, line in enumerate(lines):
            if line.startswith("def "):
                start = i
                #name就是函数名
                name = line.split("(")[0].split("def ")[1]
            if line.startswith("    return "):
                end = i
                #将函数以dict的形式进行保存
                functions[name] = lines[start:end+1]
        # organize them based on hierarchy in function names
        groupings = dict()
        for name in functions:
            parts = name.split("_")[:-1]  # last one is the constraint idx 最后一个索引是约束条件，去除
            key = "_".join(parts)
            if key not in groupings:
                groupings[key] = []
            groupings[key].append(name)
        # save them into files
        for key in groupings:
            with open(os.path.join(save_dir, f"{key}_constraints.txt"), "w") as f:
                for name in groupings[key]:
                    f.write("\n".join(functions[name]) + "\n\n")
        print(f"Constraints saved to {save_dir}")
    
    def _parse_other_metadata(self, output):
        #该方法用于从 OpenAI API 的输出中解析其他元数据，包括 num_stages、grasp_keypoints 和 release_keypoints
        data_dict = dict()
        # find num_stages
        #使用 parse.parse 函数和模板字符串 num_stages_template 从输出中解析 num_stages。
        # 如果找到了匹配的行，则将其转换为整数并存储在 data_dict 中。如果没有找到，则抛出一个 ValueError 异常
        num_stages_template = "num_stages = {num_stages}"
        for line in output.split("\n"):
            num_stages = parse.parse(num_stages_template, line)
            if num_stages is not None:
                break
        if num_stages is None:
            raise ValueError("num_stages not found in output")
        data_dict['num_stages'] = int(num_stages['num_stages'])
        # find grasp_keypoints
        #用类似的方法解析 grasp_keypoints。首先，使用模板字符串 grasp_keypoints_template 从输出中解析出 grasp_keypoints 的字符串表示。
        # 然后，将这个字符串表示转换为一个整数列表，并存储在 data_dict 中。
        grasp_keypoints_template = "grasp_keypoints = {grasp_keypoints}" #抓取关键点
        for line in output.split("\n"):
            grasp_keypoints = parse.parse(grasp_keypoints_template, line)
            if grasp_keypoints is not None:
                break
        if grasp_keypoints is None:
            raise ValueError("grasp_keypoints not found in output")
        # convert into list of ints
        grasp_keypoints = grasp_keypoints['grasp_keypoints'].replace("[", "").replace("]", "").split(",")
        grasp_keypoints = [int(x.strip()) for x in grasp_keypoints]
        data_dict['grasp_keypoints'] = grasp_keypoints
        # find release_keypoints 使用同样的方法解析 release_keypoints，并将其存储在 data_dict 中 
        release_keypoints_template = "release_keypoints = {release_keypoints}" #释放关键点
        for line in output.split("\n"):
            release_keypoints = parse.parse(release_keypoints_template, line)
            if release_keypoints is not None:
                break
        if release_keypoints is None:
            raise ValueError("release_keypoints not found in output")
        # convert into list of ints
        release_keypoints = release_keypoints['release_keypoints'].replace("[", "").replace("]", "").split(",")
        release_keypoints = [int(x.strip()) for x in release_keypoints]
        data_dict['release_keypoints'] = release_keypoints
        return data_dict

    def _save_metadata(self, metadata):
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                metadata[k] = v.tolist()
        with open(os.path.join(self.task_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {os.path.join(self.task_dir, 'metadata.json')}")

    # def generate(self, img, instruction, metadata):
    #     #这个方法用于根据给定的图像和指令生成约束条件，并将这些约束条件保存到文件中
    #     """
    #     Args:
    #         img (np.ndarray): image of the scene (H, W, 3) uint8
    #         instruction (str): instruction for the query
    #     Returns:
    #         save_dir (str): directory where the constraints
    #     """
    #     # create a directory for the task 根据时间创建一个名称
    #     #使用当前时间和指令创建一个唯一的文件名，并创建一个目录来存储任务相关的文件。
    #     fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + instruction.lower().replace(" ", "_")
    #     self.task_dir = os.path.join(self.base_dir, fname)
    #     os.makedirs(self.task_dir, exist_ok=True)
    #     # 保存查询图像
    #     image_path = os.path.join(self.task_dir, 'query_img.png')
    #     cv2.imwrite(image_path, img[..., ::-1])
    #     # 调用 _build_prompt 方法构建一个包含图像和指令的提示消息
    #     messages = self._build_prompt(image_path, instruction)
    #     # 使用 OpenAI API 的聊天补全功能，以流的方式获取响应。
    #     stream = self.client.chat.completions.create(model=self.config['model'],
    #                                                     messages=messages,
    #                                                     temperature=self.config['temperature'],
    #                                                     max_tokens=self.config['max_tokens'],
    #                                                     stream=True)
    #     output = ""
    #     start = time.time()
    #     #处理 API 返回的流数据，将其拼接成完整的输出字符串。
    #     for chunk in stream:
    #         # print(f'chunk: {chunk}')
    #         print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
    #         print(f'chunk.choices: {chunk.choices}')
    #         if chunk.choices[0].delta.content is not None:
    #             output += chunk.choices[0].delta.content
    #     print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
    #     # 保存原始输出
    #     with open(os.path.join(self.task_dir, 'output_raw.txt'), 'w') as f:
    #         f.write(output)
    #     # 调用 _parse_and_save_constraints 方法解析输出中的约束条件，并将其保存到文件中。
    #     self._parse_and_save_constraints(output, self.task_dir)
    #     # 更新元数据：
    #     metadata.update(self._parse_other_metadata(output))
    #     self._save_metadata(metadata)
    #     return self.task_dir

    def generate(self, img, instruction, metadata):
        # 这个方法用于根据给定的图像和指令生成约束条件，并将这些约束条件保存到文件中
        """
        Args:
            img (np.ndarray): image of the scene (H, W, 3) uint8
            instruction (str): instruction for the query
        Returns:
            save_dir (str): directory where the constraints
        """
        # 创建一个基于时间的任务目录
        fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + instruction.lower().replace(" ", "_")
        self.task_dir = os.path.join(self.base_dir, fname)
        os.makedirs(self.task_dir, exist_ok=True)
        
        # 保存查询图像
        image_path = os.path.join(self.task_dir, 'query_img.png')
        cv2.imwrite(image_path, img[..., ::-1])
        
        # 调用 _build_prompt 方法构建提示消息
        messages = self._build_prompt(image_path, instruction)
        
        # 使用 OpenAI API 的聊天补全功能，以流的方式获取响应
        try:
            stream = self.client.chat.completions.create(
                model=self.config['model'],
                messages=messages,
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens'],
                stream=True
            )
            output = ""
            start = time.time()
            
            # 处理 API 返回的流数据
            for chunk in stream:
                print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
                # 检查 chunk.choices 是否存在且不为空
                if hasattr(chunk, 'choices') and chunk.choices:
                    # 检查 delta.content 是否存在且不为 None
                    if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content is not None:
                        output += chunk.choices[0].delta.content
                    else:
                        print(f"Skipping chunk with no content: {chunk}")
                else:
                    print(f"Skipping empty or invalid chunk: {chunk}")
            
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
            
            # 保存原始输出
            with open(os.path.join(self.task_dir, 'output_raw.txt'), 'w') as f:
                f.write(output)
            
            # 解析并保存约束条件
            self._parse_and_save_constraints(output, self.task_dir)
            
            # 更新元数据
            metadata.update(self._parse_other_metadata(output))
            self._save_metadata(metadata)
            
            return self.task_dir

        except Exception as e:
            print(f"处理流式数据时发生错误: {e}")
            # 可选择保存错误日志到任务目录
            with open(os.path.join(self.task_dir, 'error.log'), 'w') as f:
                f.write(f"处理错误: {str(e)}")
            return None
