import os
import json
from train_network_online import train_online

class ModelImageProcessor:
    def __init__(self, user_id, order_id, sex_code, age, style_code):
        '''
        例子 
        user_id = 'O12023061322330073900001'
        order_id = 'O12023061915104587300002'
        sex_code = 100001 
        age = 25 
        style_code = '200002'  # 多个是英文逗号分隔
        '''
        self.user_id = user_id
        self.order_id = order_id
        self.sex_code = sex_code
        self.age = age
        self.style_code = style_code

    def prepare_paths(self, num_repeat="10"):
        root_path = "./train_online"
        self.model_input_path = os.path.join(root_path, self.order_id, "image")
        self.raw_input_path = os.path.join(self.model_input_path, "{}_raw".format(num_repeat))
        self.model_path = os.path.join(root_path, self.order_id, "model")
        self.log_path = os.path.join(root_path, self.order_id, "log")
        self.output_path = os.path.join(root_path, self.order_id, "output")
        os.makedirs(self.model_input_path, exist_ok=True)
        os.makedirs(self.raw_input_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        return self.raw_input_path

    def generate_prompt(self):
        positive_prompt="Ambilight, masterpiece, ultra-high quality,( ultra detailed original illustration),( 1man, upper body),(( harajuku fashion)),(( flowers with human eyes, flower eyes)), double exposure, fussion of fluid abstract art, glitch,( 2d),( original illustration composition),( fusion of limited color, maximalism artstyle, geometric artstyle, butterflies, junk art)"
        negative_prompt="easyNegative,(realistic),(3d face),(worst quality:1.2), (low quality:1.2), (lowres:1.1), (monochrome:1.1), (greyscale),(multiple legs:1.5),(extra legs:1.5),(wrong legs),(multiple hands),(missing limb),(multiple girls:1.5),garter straps,multiple heels,legwear,thghhighs,stockings,golden shoes,railing,glass"
        prompt = "{} --n {}".format(positive_prompt, negative_prompt)
        with open(os.path.join(self.output_path, "prompt.txt"), 'w') as f:
            f.write(prompt)
    
    def process(self):
        self.generate_prompt()
        # 训练模型
        output_images = train_online(self.order_id, 
                                    self.model_input_path,
                                    self.model_path,
                                    self.log_path,
                                    self.output_path)
        # 预测
        image_dict = {self.style_code: output_images}
        # 返回图片地址字典
        return image_dict

    def process_test(self) :
        '''流程测试用'''
        return {"20001":["images/O12023061915104587300002/output/20001_O12023061915104587300001_1.jpg",
                    "images/O12023061915104587300002/output/20001_O12023061915104587300001_2.jpg"],
        "20002":["images/O12023061915104587300002/output/20002_O12023061915104587300001_1.jpg",
                    "images/O12023061915104587300002/output/20002_O12023061915104587300001_2.jpg"]}



# processor = ModelImageProcessor(user_id, order_id, sex_code, age, style_code, input_photo_files)
# result    = processor.process()

# print(result)

