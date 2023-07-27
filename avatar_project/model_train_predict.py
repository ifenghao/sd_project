import os
import json
from jinja2 import Template
from collections import OrderedDict
from train_network_online import train_online
from gen_img_diffusers_online import gen_img

class ModelImageProcessor:
    def __init__(self, logger, user_id, order_id, sex_code, age, style_code):
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
        self.logger = logger

    def prepare_paths(self, num_repeat="5"):
        root_path = "./train_online"
        self.image_recieve_path = os.path.join(root_path, self.order_id, "image_raw")
        self.model_input_path = os.path.join(root_path, self.order_id, "image")
        self.image_crop_path = os.path.join(root_path, self.order_id, "image", "{}_crop".format(num_repeat))
        self.model_path = os.path.join(root_path, self.order_id, "model")
        self.log_path = os.path.join(root_path, self.order_id, "log")
        self.output_path = os.path.join(root_path, self.order_id, "output")
        os.makedirs(self.image_recieve_path, exist_ok=True)
        os.makedirs(self.image_crop_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        return self.image_recieve_path, self.image_crop_path

    def train(self, gen_sample_image=True, params={}):
        if not gen_sample_image:
            params['sample_every_n_steps'] = None
            params['sample_every_n_epochs'] = None
        # 训练模型
        try:
            output_sample_images = train_online(
                self.order_id, 
                self.model_input_path,
                self.model_path,
                self.log_path,
                self.output_path,
                **params)
        except Exception as e:
            self.logger.error('order_id:{},训练出错 {}'.format(self.order_id, e))
            output_sample_images = []
        return output_sample_images
    
    def generate(self, style_res_list, images_per_prompt, use_step=-1, highres_fix=False, params={}):
        model_file_list = os.listdir(self.model_path)
        model_file_list = list(filter(lambda t: t.startswith(self.order_id), model_file_list))
        total_steps = len(model_file_list)
        if total_steps == 0 or use_step >= total_steps:
            self.logger.error('order_id:{},No lora model is loaded'.format(self.order_id))
            return []
        model_file = os.path.join(self.model_path, model_file_list[use_step])
        self.logger.info('order_id:{},select model file: {}'.format(self.order_id, model_file))
        # 分ckpt生成
        gender_des, gender, age_des = preprocess_gender_age(self.sex_code, self.age, reverse_gender=False)
        gen_img_pass_list  = parse_gen_info(style_res_list, gender_des, gender, age_des)
        # 生成图片
        lora_path = './models/lora/'
        output_gen_images = []
        output_style_codes = []
        for params in gen_img_pass_list:
            network_weights_paths = [model_file if name == 'train.safetensors' else lora_path + name for name in params['network_weights']]
            try:
                output_pass_images = gen_img(outdir=self.output_path,
                                            network_weights=network_weights_paths,
                                            ckpt=params['ckpt'],
                                            prompt=params['prompt'],
                                            vae=params['vae'],
                                            highres_fix=highres_fix,
                                            images_per_prompt=images_per_prompt)
            except Exception as e:
                self.logger.error('order_id:{},生成出错 {}'.format(self.order_id, e))
                output_pass_images = []
            output_gen_images.extend(output_pass_images)
            output_style_codes.extend(params['style_code'])
        return output_style_codes, output_gen_images

    def process_with_gen(self, style_res_list, images_per_prompt, gen_sample_image=True, use_step=-1, highres_fix=False, train_params={}, gen_params={}):
        if len(style_res_list) == 0:
            self.logger.info('order_id:{},没有风格用于生成'.format(self.order_id))
            return {}
        output_sample_images = self.train(gen_sample_image, params=train_params)
        output_style_codes, output_gen_images = self.generate(style_res_list, images_per_prompt, use_step, highres_fix, params=gen_params)
        output_images = output_gen_images
        # 整理图片结果
        num_style = len(output_style_codes)
        style_image_list = []
        for i in range(num_style):
            style_image_list.append([])
        for i, image in enumerate(output_images):
            style_image_list[i // images_per_prompt].append(image)
        # 返回图片地址字典
        image_dict = dict(zip(output_style_codes, style_image_list))
        return image_dict

    def process_test(self) :
        '''流程测试用'''
        return {"20001":["images/O12023061915104587300002/output/20001_O12023061915104587300001_1.jpg",
                    "images/O12023061915104587300002/output/20001_O12023061915104587300001_2.jpg"],
        "20002":["images/O12023061915104587300002/output/20002_O12023061915104587300001_1.jpg",
                    "images/O12023061915104587300002/output/20002_O12023061915104587300001_2.jpg"]}


def preprocess_gender_age(sex_code, age, reverse_gender=False):
    '''处理性别和年龄,默认不性别反转:
       - {{age_des}} : 'girl'/'young woman'/'woman'/'boy'/'young man'/'man'
       - {{gender}}  : 'female'/'male' 
       - {{age_des}} : f'{age} years old'
       *后期放到后端处理
    '''
    def get_age_index(age, age_range):
        for index, ar in enumerate(age_range):
            if age <= ar:
                return index
        return len(age_range)
    sex_dict = {
        '100001': {'gender': 'male', 'obj_list': ['boy', 'young man', 'man'], 'age_range': [16, 28]},
        '100002': {'gender': 'female', 'obj_list': ['girl', 'young woman', 'woman'], 'age_range': [26, 40]}
    }
    sex_info = sex_dict.get(str(sex_code))
    if sex_info:
        if reverse_gender:
            gender_des = sex_info['obj_list'][get_age_index(int(age), sex_info['age_range'])]
            gender = 'female' if sex_info['gender'] == 'male' else 'male'
        else:
            gender_des = sex_info['obj_list'][get_age_index(int(age), sex_info['age_range'])]
            gender = sex_info['gender']
    age_des = f'{age} years old'
    return gender_des, gender, age_des


def modified_prompt(prompt_template,gender_des,gender,age_des):
    '''根据预处理好的年龄、性别修改prompt中的参数'''
    template = Template(prompt_template)
    modified_prompt = template.render(gender_des=gender_des, gender=gender, age_des=age_des)
    return modified_prompt


def parse_gen_info(style_infos, gender_des, gender, age_des):
    def style_params_to_prompt(style_params, network_mul):
        prompt_res = []
        positive_prompt = style_params.get('posPrompt')
        if positive_prompt is not None:
            prompt_res.append(modified_prompt(positive_prompt, gender_des, gender, age_des))
        negative_prompt = style_params.get('negPrompt')
        if negative_prompt is not None:
            prompt_res.append('--n {}'.format(modified_prompt(negative_prompt, gender_des, gender, age_des)))
        prompt_res.append('--w {}'.format(style_params.get('width', 512)))
        prompt_res.append('--h {}'.format(style_params.get('height', 512)))
        prompt_res.append('--s {}'.format(style_params.get('steps', 30)))
        prompt_res.append('--sp {}'.format(style_params.get('sampler', 'euler_a')))
        seed = style_params.get('seed')
        if seed is not None and seed != 'None':
            prompt_res.append('--d {}'.format(seed))
        prompt_res.append('--l {}'.format(style_params.get('scale', 7)))
        negative_scale = style_params.get('negative_scale')
        if negative_scale is not None and negative_scale != 'None':
            prompt_res.append('--nl {}'.format(negative_scale))
        if network_mul:
            prompt_res.append('--am {}'.format(','.join(network_mul)))
        return ' '.join(prompt_res)

    # 分ckpt解析需要加载lora
    ckpt_dict = OrderedDict()    
    for style in style_infos:
        ckpt = style.get('ckpt', None)
        if ckpt is None:
            print('No ckpt assigned for style code {}'.format(style.get('code', '')))
            continue
        if ckpt not in ckpt_dict:
            ckpt_dict[ckpt] = {'ckpt_styles': [], 'lora_index': {}, 'lora_num': 0, 'vae': style.get('vae', 'None')}
        ckpt_dict[ckpt]['ckpt_styles'].append(style)
        if ckpt_dict[ckpt]['vae'] == 'None':
            ckpt_dict[ckpt]['vae'] = style.get('vae', 'None')
        lora_list = style.get('network_weights', [])
        for lora in lora_list:
            if lora not in ckpt_dict[ckpt]['lora_index']:
                ckpt_dict[ckpt]['lora_index'][lora] = ckpt_dict[ckpt]['lora_num']
                ckpt_dict[ckpt]['lora_num'] += 1

    # 分ckpt分配lora权重
    gen_img_pass_list = []
    for ckpt, ckpt_info in ckpt_dict.items():
        lora_index = ckpt_info['lora_index']
        lora_num = ckpt_info['lora_num']
        network_weights = [''] * lora_num
        for lora, index in lora_index.items():
            network_weights[index] = lora

        vae = None if ckpt_info['vae'] == 'None' else ckpt_info['vae']
        prompt_list = []
        style_code_list = []
        for ckpt_style in ckpt_info['ckpt_styles']:
            network_mul = ['0.0'] * lora_num
            weights_list = ckpt_style.get('network_weights', [])
            mul_list = ckpt_style.get('network_mul', [])
            for weights, mul in zip(weights_list, mul_list):
                network_mul[lora_index[weights]] = str(mul)
            prompt_list.append(style_params_to_prompt(ckpt_style, network_mul))
            style_code_list.append(ckpt_style.get('code'))
        gen_img_pass_list.append({'ckpt': ckpt, 'network_weights': network_weights, 'prompt': '\n'.join(prompt_list), 'vae': vae, 'style_code': style_code_list})
    return gen_img_pass_list
