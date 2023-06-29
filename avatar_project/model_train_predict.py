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

    def prepare_paths(self, num_repeat="30"):
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
        sex_str, age_str = generate_info_prompt(self.sex_code, self.age)
        prompt_dict = generate_prompt_dict(sex_str, age_str)
        prompt_list = []
        style_code_list = str(self.style_code).split(',')
        valid_style_code_list = []
        for style_code in style_code_list:
            if style_code in prompt_dict:
                prompt = prompt_dict[style_code]
                prompt_list.append("{} --n {}\n".format(prompt['pos'], prompt['neg']))
                valid_style_code_list.append(style_code)
        if len(valid_style_code_list) == 0:
            return []
        with open(os.path.join(self.output_path, "prompt.txt"), 'w') as f:
            f.writelines(prompt_list)
        return valid_style_code_list
    
    def process(self, logger):
        valid_style_code_list = self.generate_prompt()
        if len(valid_style_code_list) == 0:
            logger.info('order_id:{},没有风格用于生成'.format(self.order_id))
            return []
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
        # 整理图片结果
        num_style = len(valid_style_code_list)
        style_image_list = []
        for i in range(num_style):
            style_image_list.append([])
        for i, image in enumerate(output_images):
            style_image_list[i % num_style].append(image)
        # 返回图片地址字典
        image_dict = dict(zip(valid_style_code_list, style_image_list))
        return image_dict

    def process_test(self) :
        '''流程测试用'''
        return {"20001":["images/O12023061915104587300002/output/20001_O12023061915104587300001_1.jpg",
                    "images/O12023061915104587300002/output/20001_O12023061915104587300001_2.jpg"],
        "20002":["images/O12023061915104587300002/output/20002_O12023061915104587300001_1.jpg",
                    "images/O12023061915104587300002/output/20002_O12023061915104587300001_2.jpg"]}


def generate_info_prompt(sex_code, age):
    def get_age_index(age, age_range):
        index = 0
        age = int(age)
        for ar in age_range:
            if age <= ar:
                return index
            index += 1
        return index

    sex_str = 'human'
    age_str = '{age} year old'.format(age=age)
    male_obj_list = ['boy', ' young man', 'man']
    male_age_list = [16, 28]
    female_obj_list = ['girl', ' young woman', 'woman']
    female_age_list = [26, 40]

    if int(sex_code) == 100001:
        sex_str = male_obj_list[get_age_index(age, male_age_list)]
    elif int(sex_code) == 100002:
        sex_str = female_obj_list[get_age_index(age, female_age_list)]
    return sex_str, age_str


def generate_prompt_dict(sex_str, age_str):
    prompt_dict = {
        '200001': {
            'pos': "3dmm style,(masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2), (fractal art:1.3), 1{sex}, beautiful, high detailed, purple hair with a hint of pink, pink eyes, dark lighting, serious face, looking the sky, sky, medium shot, black sweater, jewelry, {age}".format(sex=sex_str, age=age_str),
            'neg': "tattooing,Neck decoration, collar, necklace,collar,badhandv4, paintings, sketches, (worst qualit:2), (low quality:2), (normal quality:2), lowers, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), manboobs, (backlight:1.2), double navel, muted arms, hused arms, neck lace, analog, analog effects, (sunglass:1.4), nipples, nsfw, bad architecture, watermark, (mole:1.5), EasyNegative"
        },
        '200002': {
            'pos': "Ambilight, masterpiece, ultra-high quality,( ultra detailed original illustration),( 1{sex}, upper body),(( harajuku fashion)),(( flowers with human eyes, flower eyes)), double exposure, fussion of fluid abstract art, glitch,( 2d),( original illustration composition),( fusion of limited color, maximalism artstyle, geometric artstyle, butterflies, junk art), {age}".format(sex=sex_str, age=age_str),
            'neg': "easyNegative,(realistic),(3d face),(worst quality:1.2), (low quality:1.2), (lowres:1.1), (monochrome:1.1), (greyscale),(multiple legs:1.5),(extra legs:1.5),(wrong legs),(multiple hands),(missing limb),(multiple bodies:1.5),garter straps,multiple heels,legwear,thghhighs,stockings,golden shoes,railing,glass"
        },
    }
    return prompt_dict
