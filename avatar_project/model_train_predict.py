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
        sex_str = 'man'
        if int(self.sex_code) == 100001:
            sex_str = 'boy'
        elif int(self.sex_code) == 100002:
            sex_str = 'girl'

        prompt_dict = generate_prompt_dict(sex_str)
        prompt_list = []
        style_code_list = str(self.style_code).split(',')
        for style_code in style_code_list:
            if style_code in prompt_dict:
                prompt = prompt_dict[style_code]
                prompt_list.append("{} --n {}\n".format(prompt['pos'], prompt['neg']))
        with open(os.path.join(self.output_path, "prompt.txt"), 'w') as f:
            f.writelines(prompt_list)
    
    def process(self, logger):
        self.generate_prompt()
        # 训练模型
        try:
            output_images = train_online(self.order_id, 
                                        self.model_input_path,
                                        self.model_path,
                                        self.log_path,
                                        self.output_path)
        except Exception as e:
            logger.error('order_id:{},执行出错 {}'.format(self.order_id, e))
            output_images = []
        # 预测
        image_dict = {self.style_code.split(',')[0]: output_images}
        # 返回图片地址字典
        return image_dict

    def process_test(self) :
        '''流程测试用'''
        return {"20001":["images/O12023061915104587300002/output/20001_O12023061915104587300001_1.jpg",
                    "images/O12023061915104587300002/output/20001_O12023061915104587300001_2.jpg"],
        "20002":["images/O12023061915104587300002/output/20002_O12023061915104587300001_1.jpg",
                    "images/O12023061915104587300002/output/20002_O12023061915104587300001_2.jpg"]}


def generate_prompt_dict(sex_str):
    prompt_dict = {
        '200001': {
            'pos': "3dmm style,(masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2), (fractal art:1.3), 1{sex}, beautiful, high detailed, purple hair with a hint of pink, pink eyes, dark lighting, serious face, looking the sky, sky, medium shot, black sweater, jewelry".format(sex=sex_str),
            'neg': "tattooing,Neck decoration, collar, necklace,collar,badhandv4, paintings, sketches, (worst qualit:2), (low quality:2), (normal quality:2), lowers, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), manboobs, (backlight:1.2), double navel, muted arms, hused arms, neck lace, analog, analog effects, (sunglass:1.4), nipples, nsfw, bad architecture, watermark, (mole:1.5), EasyNegative"
        },
        '200002': {
            'pos': "Ambilight, masterpiece, ultra-high quality,( ultra detailed original illustration),( 1{sex}, upper body),(( harajuku fashion)),(( flowers with human eyes, flower eyes)), double exposure, fussion of fluid abstract art, glitch,( 2d),( original illustration composition),( fusion of limited color, maximalism artstyle, geometric artstyle, butterflies, junk art)".format(sex=sex_str),
            'neg': "easyNegative,(realistic),(3d face),(worst quality:1.2), (low quality:1.2), (lowres:1.1), (monochrome:1.1), (greyscale),(multiple legs:1.5),(extra legs:1.5),(wrong legs),(multiple hands),(missing limb),(multiple {sex}s:1.5),garter straps,multiple heels,legwear,thghhighs,stockings,golden shoes,railing,glass".format(sex=sex_str)
        },
    }
    return prompt_dict
