import os
import sys
import shutil
import dlib
import cv2
from PIL import Image
from collections import OrderedDict
from jinja2 import Template
from train_network_online import train_online
from tag_images_by_wd14_tagger_online import tag_images
from gen_img_diffusers_online import gen_img

class ModelPreprocessing:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def crop_face_from_path(self, input_path, crop_path, scales=[0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8]) : 
        crop_num = 0 
        for root, dirs, files in os.walk(input_path):
            for file_name in files: 
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path    = os.path.join(root, file_name)
                    for scale in scales:
                        cropped_image = self.extract_square_face_and_shoulders(image_path, scale=scale) 
                        scale_crop_path = crop_path + str(scale)
                        if cropped_image is not None : 
                            crop_num +=1
                            os.makedirs(scale_crop_path, exist_ok=True) 
                            self.save_cropped_image(os.path.join(scale_crop_path, file_name),  cropped_image) 
                        else :
                            self.copy_image_to_folder(scale_crop_path, image_path)   
        return crop_num
    
    def crop_face_from_path_auto_scale(self, input_path, crop_path, scales_num=10, min_scale=0.8) : 
        copy_num = 0
        crop_num = 0
        for root, dirs, files in os.walk(input_path):
            for file_name in files: 
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path    = os.path.join(root, file_name)
                    image, face = self.get_face_detect_result(image_path)
                    if face is None:
                        self.copy_image_to_folder(crop_path, image_path)
                        copy_num += 1
                    else:
                        max_scale = self.get_max_face_scale(image, face)
                        scale_step = (max_scale - min_scale) / scales_num
                        for i in range(scales_num):
                            cropped_image = self.get_square_face_at_scale(image, face, min_scale + i * scale_step)
                            scale_crop_path = crop_path + str(i)
                            os.makedirs(scale_crop_path, exist_ok=True)
                            self.save_cropped_image(os.path.join(scale_crop_path, file_name),  cropped_image)
                            crop_num +=1 
        return crop_num

    def extract_face_and_shoulders(self, image_path, scale_x=2.8, scale_y=3):
        try : 
            image = cv2.imread(image_path) 
            gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) != 1 : 
                return None 
            else : 
                face = faces[0] 
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                padding_x = int(w * (scale_x - 1) / 2)
                padding_y = int(h * (scale_y - 1) / 2)
                # 扩展裁剪区域
                x -= padding_x
                y -= padding_y
                w += 2 * padding_x
                h += 2 * padding_y
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                # 裁剪出脸部和肩膀区域
                cropped_face = image[y:y+h, x:x+w]
                return cropped_face
        except Exception as e:
            print(f"截取头像失败:{e},{image_path}")  
            return None 
        
    def extract_square_face_and_shoulders(self, image_path, scale=3):
        try : 
            image = cv2.imread(image_path) 
            gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) != 1 : 
                return None 
            else : 
                face = faces[0] 
                left, top, w, h = face.left(), face.top(), face.width(), face.height()
                side = max(w, h)
                # 扩展裁剪区域
                padding = int(side * (scale - 1) / 2)
                left_bound = left - padding
                top_bound = top - padding
                right_bound = left + side + padding
                bottom_bound = top + side + padding
                max_diff = 0
                if left_bound < 0:
                    max_diff = max(max_diff, -left_bound)
                if top_bound < 0:
                    max_diff = max(max_diff, -top_bound)
                if right_bound > image.shape[1] - 1:
                    max_diff = max(max_diff, right_bound - image.shape[1] + 1)
                if bottom_bound > image.shape[0] - 1:
                    max_diff = max(max_diff, bottom_bound - image.shape[0] + 1)
                left_bound += max_diff
                top_bound += max_diff
                right_bound -= max_diff
                bottom_bound -= max_diff
                # 裁剪出脸部和肩膀区域
                cropped_face = image[top_bound:bottom_bound, left_bound:right_bound]
                return cropped_face
        except Exception as e:
            print(f"截取头像失败:{e},{image_path}")  
            return None 
        
    def get_face_detect_result(self, image_path, min_face_reso=32):
        try : 
            image = cv2.imread(image_path) 
            gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) != 1 : 
                return None, None 
            else:
                face = faces[0]
                w, h = face.width(), face.height()
                if w <= min_face_reso and h <= min_face_reso:
                    return None, None
                return image, face
        except Exception as e:
            print(f"截取头像失败:{e},{image_path}")
            return None, None
        
    def get_max_face_scale(self, image, face):
        left, top, w, h = max(face.left(), 0), max(face.top(), 0), face.width(), face.height()
        image_width, image_height = image.shape[1], image.shape[0]
        side = max(w, h)
        left_max_pad = left
        right_max_pad = image_width - 1 - (left + side)
        top_max_pad = top
        bottom_max_pad = image_height - 1 - (top + side)
        max_pad_min = min(left_max_pad, right_max_pad, top_max_pad, bottom_max_pad)
        max_scale = 2 * max_pad_min / side + 1
        return max_scale
    
    def get_square_face_at_scale(self, image, face, scale):
        left, top, w, h = max(face.left(), 0), max(face.top(), 0), face.width(), face.height()
        side = max(w, h)
        # 扩展裁剪区域
        padding = int(side * (scale - 1) / 2)
        left_bound = left - padding
        top_bound = top - padding
        right_bound = left + side + padding
        bottom_bound = top + side + padding
        cropped_face = image[top_bound:bottom_bound, left_bound:right_bound]
        return cropped_face
    
    def copy_image_from_path(self, input_path, copy_path):
        copy_num = 0
        for root, dirs, files in os.walk(input_path):
            for file_name in files: 
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    copy_num += 1
                    image_path    = os.path.join(root, file_name)
                    self.copy_image_to_folder(copy_path, image_path)
        return copy_num

    def copy_image_to_folder(self, destination_folder, source_file):
        try:        
            shutil.copy2(source_file, destination_folder)
        except Exception as e:
            print(f"Error copying file:{e}") 

    def save_cropped_image(self, cropped_image_path, cropped_image): 
        try:
            cv2.imwrite(cropped_image_path, cropped_image,[cv2.IMWRITE_JPEG_QUALITY, 100])
        except Exception as e:
            print(f"Error saving cropped image: {e}")


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

    def prepare_paths(self, raw_path, num_repeat="5"):
        root_path = "./train"
        self.image_recieve_path = os.path.join(raw_path, self.order_id)
        self.model_input_path = os.path.join(root_path, self.order_id, "image")
        self.image_crop_path = os.path.join(root_path, self.order_id, "image", "{}_crop".format(num_repeat))
        self.model_path = os.path.join(root_path, self.order_id, "model")
        self.log_path = os.path.join(root_path, self.order_id, "log")
        self.output_path = os.path.join(root_path, self.order_id, "output")
        os.makedirs(self.image_crop_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)                 
        return self.image_recieve_path, self.image_crop_path
    
    def generate_tags(self, batch_size=1,
                    max_data_loader_n_workers=16,
                    general_threshold=0.35,
                    character_threshold=0.5):
        tag_images(train_data_dir=self.model_input_path,
                   model_dir='./models/wd14_tagger_model',
                   class_tag=self.order_id,
                   batch_size=batch_size,
                   max_data_loader_n_workers=max_data_loader_n_workers,
                   general_threshold=general_threshold,
                   character_threshold=character_threshold)
    
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
            print('order_id:{},训练出错 {}'.format(self.order_id, e))
            output_sample_images = []
        return output_sample_images
    
    def generate(self, style_res_list, images_per_prompt, use_step=-1, highres_fix=False, params={}):
        model_file_list = os.listdir(self.model_path)
        model_file_list = list(filter(lambda t: t.startswith(self.order_id), model_file_list))
        total_steps = len(model_file_list)
        if total_steps == 0 or use_step >= total_steps:
            print('No lora model is loaded')
            return []
        model_file = os.path.join(self.model_path, model_file_list[use_step])
        print('select model file: ' + model_file)
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
                print('order_id:{},生成出错 {}'.format(self.order_id, e))
                output_pass_images = []
            output_gen_images.extend(output_pass_images)
            output_style_codes.extend(params['style_code'])
        return output_style_codes, output_gen_images
    
    def process(self, style_res_list, images_per_prompt, run_train=True, gen_sample_image=True, use_step=-1, highres_fix=False, train_params={}, gen_params={}):
        if run_train:
            output_sample_images = self.train(gen_sample_image, params=train_params)
        output_style_codes, output_gen_images = self.generate(style_res_list, images_per_prompt, use_step, highres_fix, params=gen_params)
        return len(output_style_codes), output_gen_images

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


def generate_prompt():
    prompt_list = []
    prompt_list.append(
        {
            'code': '200001',
            'posPrompt': " (wearing equitation_outfit, wearing cap:1.3), white equitation_outfit, good hand,4k, high-res, masterpiece, best quality, head:1.3,((Hasselblad photography)), finely detailed skin, sharp focus, (cinematic lighting), night, soft lighting, dynamic angle, [:(detailed face:1.2):0.2], medium breasts,(((in front of a crowd))), outside",
            'negPrompt': "NG_DeepNagetive_V1_75T,(greyscale:1.2),naked, nude, nipples, vagina, nsfw, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans",
            'ckpt': 'dreamshaper_631BakedVae.safetensors',
            'network_weights': ['train.safetensors'],
            'network_mul': [1.0],
            'scale': 7,
            'negative_scale': None,
            'seed': None,
            'sampler': 'euler_a',
            'steps': 30,
            'width': 512,
            'height': 768
        },
    )
    prompt_list.append(
        {
            'code': '200002',
            'posPrompt': "mj3d style,3dmm,3d,(masterpiece, best quality:1.1), elf, light blue hair, glasses, mole on mouth ,anime , (smile:0.5), 1{{gender_des}}, upper body, {{age_des}}",
            'negPrompt': "NG_DeepNagetive_V1_75T,(greyscale:1.2),naked, nude, nipples, vagina, nsfw, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans",
            'ckpt': 'dreamshaper_631BakedVae.safetensors',
            'network_weights': ['train.safetensors'],
            'network_mul': [1.0],
            'scale': 7,
            'negative_scale': None,
            'seed': None,
            'sampler': 'euler_a',
            'steps': 30,
            'width': 512,
            'height': 768
        },
    )
    prompt_list.append(
        {
            'code': '200003',
            'posPrompt': "8k portrait of beautiful cyborg with brown hair, intricate, elegant, highly detailed, majestic, digital photography, art by artgerm and ruan jia and greg rutkowski surreal painting gold butterfly filigree, broken glass, (masterpiece, sidelighting, finely detailed beautiful eyes: 1.2), hdr, 1{{gender_des}}, upper body, {{age_des}}",
            'negPrompt': "NG_DeepNagetive_V1_75T,(greyscale:1.2),naked, nude, nipples, vagina, nsfw, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans",
            'ckpt': 'dreamshaper_631BakedVae.safetensors',
            'network_weights': ['train.safetensors'],
            'network_mul': [1.0],
            'scale': 7,
            'negative_scale': None,
            'seed': None,
            'sampler': 'euler_a',
            'steps': 30,
            'width': 512,
            'height': 768
        },
    )
    prompt_list.append(
        {
            'code': '200004',
            'posPrompt': "((master piece)),best quality, illustration, 1{{gender_des}}, upper body, Look out the window, beautiful detailed eyes, (beautiful detailed cyberpunk city), beautiful detailed hair, {{age_des}}",
            'negPrompt': "NG_DeepNagetive_V1_75T,(greyscale:1.2),naked, nude, nipples, vagina, nsfw, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans",
            'ckpt': 'dreamshaper_631BakedVae.safetensors',
            'network_weights': ['train.safetensors'],
            'network_mul': [1.0],
            'scale': 7,
            'negative_scale': None,
            'seed': None,
            'sampler': 'euler_a',
            'steps': 30,
            'width': 512,
            'height': 768
        }
    )
    prompt_list.append(
        {
            'code': '200005',
            'posPrompt': "Ambilight, masterpiece, ultra-high quality,( ultra detailed original illustration),( 1{{gender_des}}, upper body),(( harajuku fashion)),(( flowers with human eyes, flower eyes)), double exposure, fussion of fluid abstract art, glitch,( 2d),( original illustration composition),( fusion of limited color, maximalism artstyle, geometric artstyle, butterflies, junk art), {{age_des}}",
            'negPrompt': "NG_DeepNagetive_V1_75T,(greyscale:1.2),naked, nude, nipples, vagina, nsfw, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans",
            'ckpt': 'dreamshaper_631BakedVae.safetensors',
            'network_weights': ['train.safetensors'],
            'network_mul': [1.0],
            'scale': 7,
            'negative_scale': None,
            'seed': None,
            'sampler': 'euler_a',
            'steps': 30,
            'width': 512,
            'height': 768
        },
    )
    prompt_list.append(
        {
            'code': '200006',
            'posPrompt': "((master piece)),best quality, illustration, dark, 1{{gender_des}}, upper body, In the wilderness,High mountain,Snow-capped mountains in the distance, castle, beautiful detailed eyes, beautiful detailed hair, {{age_des}}",
            'negPrompt': "NG_DeepNagetive_V1_75T,(greyscale:1.2),naked, nude, nipples, vagina, nsfw, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans",
            'ckpt': 'dreamshaper_631BakedVae.safetensors',
            'network_weights': ['train.safetensors'],
            'network_mul': [1.0],
            'scale': 7,
            'negative_scale': None,
            'seed': None,
            'sampler': 'euler_a',
            'steps': 30,
            'width': 512,
            'height': 768
        },
    )
    prompt_list.append(
        {
            'code': '200007',
            'posPrompt': "1{{gender_des}}, solo, long hair, looking at viewer, colorful background, colorful hair, simple background, colorful eyes, lips, closed mouth, ribbon, hair ribbon, bangs,school uniform, upper body, parted bangs, colorful ribbon , nose",
            'negPrompt': "NG_DeepNagetive_V1_75T,(greyscale:1.2),naked, nude, nipples, vagina, nsfw, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans",
            'ckpt': 'revAnimated_v122.safetensors',
            'network_weights': ['train.safetensors','3DMM_V12.safetensors','more_details.safetensors'],
            'network_mul': [1.0,0.75,0.1],
            'scale': 7,
            'negative_scale': None,
            'seed': None,
            'sampler': 'euler_a',
            'steps': 25,
            'width': 512,
            'height': 768
        },
    )
    prompt_list.append(
        {
            'code': '200009',
            'posPrompt': "1{{gender_des}}, ((delicate skin)),mechanical collar,abstract art, half cyberpunk machine melting into human face, beautiful, colorful paint skin, bobcut, portrait, extreme detail, (colorful background:1.2), color splash,Neon city, RAW candid cinema, 16mm, color graded portra 400 film, remarkable color, ultra realistic, remarkable detailed pupils, shot with cinematic camera, 8K",
            'negPrompt': "easynegative, CyberRealistic_Negative-neg, ng_deepnegative_v1_75t",
            'ckpt': 'majicmixRealistic_v6.safetensors',
            'network_weights': ['train.safetensors','more_details.safetensors'],
            'network_mul': [1.0,0.5],
            'scale': 7,
            'negative_scale': None,
            'seed': None,
            'sampler': 'euler_a',
            'steps': 27,
            'width': 512,
            'height': 768
        },
    )
    prompt_list.append(
        {
            'code': '200010',
            'posPrompt': "best quality,masterpiece,highres,1{{gender_des}},blush,(seductive smile:0.8),star-shaped pupils,red china hanfu,hanfu,chinese clothes,hair ornament,necklace,jewelry,Beautiful face,upon_body,tyndall effect,photorealistic,dark studio,rim lighting,two tone lighting,(high detailed skin:1.2),8k uhd,dslr,soft lighting,high quality,volumetric lighting,candid,Photograph,high resolution,4k,8k,Bokeh",
            'negPrompt': "(((simple background))),monochrome,lowres,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry,lowres,bad anatomy,bad hands,text,error,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry,ugly,pregnant,vore,duplicate,morbid,mut ilated,tran nsexual,hermaphrodite,long neck,mutated hands,poorly drawn hands,poorly drawn face,mutation,deformed,blurry,bad anatomy,bad proportions,malformed limbs,extra limbs,cloned face,disfigured,gross proportions,(((missing arms))),(((missing legs))),(((extra arms))),(((extra legs))),pubic hair,plump,bad legs,error legs,username,blurry,bad feet, badhandv4, CyberRealistic_Negative-neg, EasyNegative, ng_deepnegative_v1_75t",
            'ckpt': '3Guofeng3_v32Light.safetensors',
            'network_weights': ['train.safetensors','more_details.safetensors'],
            'network_mul': [1.0,0.6],
            'scale': 7,
            'negative_scale': None,
            'seed': None,
            'sampler': 'euler',
            'steps': 20,
            'width': 512,
            'height': 768
        },
    )
    prompt_list.append(
        {
            'code': '200011',
            'posPrompt': "masterpiece, best quality,RAW,(expensive portrait of a {{gender_des}}), gorgeous strapless evening gown,exquisite necklace,exquisite earrings,delicate skin,detailed hair, perfect face, beautiful face, detailed eyes,smiling eyes,smiling,beautiful eyelashes,looking at viewer, (((half-length portrait))), professional lighting, photography studio,artistic black background, god rays, artistic photography, detailed face, (body towards viewer),night,light on face,volumetric lighting,tyndall effect,rim lighting,Bokeh,DSLR",
            'negPrompt': "(worst quality, low quality, bad_pictures, negative_hand-neg:1.2)(large breasts:1.3),EasyNegative, ng_deepnegative_v1_75t, CyberRealistic_Negative-neg",
            'ckpt': 'leosamsMoonfilm_filmGrain20.safetensors',
            'network_weights': ['train.safetensors','more_details.safetensors'],
            'network_mul': [1.0,0.8],
            'scale': 7,
            'negative_scale': None,
            'seed': None,
            'sampler': 'euler',
            'steps': 25,
            'width': 512,
            'height': 768
        },
    )
    prompt_list.append(
        {
            'code': '200012',
            'posPrompt': "Best portrait photography, 35mm film, natural blurry, 1{{gender_des}}, sun dress, wide brimmed hat, radiant complexion, whimsical pose, fluttering hair, golden sunlight, macro shot, shallow depth of field, bokeh, dreamy",
            'negPrompt': "(worst quality, low quality, bad_pictures, negative_hand-neg:1.2)(large breasts:1.3),EasyNegative, ng_deepnegative_v1_75t, CyberRealistic_Negative-neg",
            'ckpt': 'leosamsMoonfilm_filmGrain20.safetensors',
            'network_weights': ['train.safetensors','more_details.safetensors'],
            'network_mul': [1.0,0.8],
            'scale': 7,
            'negative_scale': None,
            'seed': None,
            'sampler': 'euler',
            'steps': 20,
            'width': 512,
            'height': 768
        }
    )
    prompt_list.append(
        {
            'code': '200013',
            'posPrompt': "8k portrait of {{gender}} celestial, 1{{gender_des}}, Style-Gravitymagic, sparkle, light particles, halo, looking at viewer, bioluminescent flame, bioluminescence, phoenix, beautiful eyes, (upper body:1.2), Vibrant, Colorful, Color, 8k, high quality, hyper realistic, professional photography, {{age_des}}",
            'negPrompt': "BadDream, FastNegativeV2, naked, nude, nipples, vagina, glans",
            'ckpt': 'dreamshaper_7.safetensors',
            'network_weights': ['train.safetensors'],
            'network_mul': [1.0],
            'scale': 9,
            'negative_scale': None,
            'seed': None,
            'sampler': 'euler_a',
            'steps': 30,
            'width': 512,
            'height': 768
        },
    )
    prompt_list.append(
        {
            'code': '200014',
            'posPrompt': "Style-PaintMagic, photo of a beautiful goth girl with thick flowing (liquid paint rainbow hair:1.1) made of paint and defies gravity, space background, highly detailed, intricate, amazing, trending, paint splatter, paint drops",
            'negPrompt': "bad anatomy, low-res, (watermarks:1.2), username, paintings, sketches, (worst quality:2), hat, hats,(low quality:2), (normal quality:2), monochrome, grayscale, (easynegative:1.1), [ng_deepnegative_v1_75t] bad anatomy, low-res, poorly drawn face, disfigured hands, poorly drawn eyebrows, bad body perspective, animal tail, anime, nipples, pussy, wrong anatomy, poorly drawn legs, wrong perspective legs, poorly drawn hands, (bad-hands-5:1.8), wrong hand, yellow light, canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], signature, watermarks, vile_prompt3",
            'ckpt': 'dreamshaper_7.safetensors',
            'network_weights': ['train.safetensors'],
            'network_mul': [1.0],
            'scale': 9,
            'negative_scale': None,
            'seed': None,
            'sampler': 'euler_a',
            'steps': 30,
            'width': 512,
            'height': 768
        },
    )
    prompt_list.append(
        {
            'code': '200015',
            'posPrompt': "(2D),Ambilight, masterpiece, best quality,realistic photography,detailed asian face, (original illustration composition),1{{gender_des}},in car background,black business suit,(upper body without hands)",
            'negPrompt': "(naked),easyNegative,(realistic),(3d face),(worst quality:1.2), (low quality:1.2), (lowres:1.1), (monochrome:1.1), (greyscale),(multiple legs:1.5),(extra legs:1.5),(wrong legs),(multiple hands),(missing limb),(multiple bodies:1.5),garter straps,multiple heels,legwear,thghhighs,stockings,golden shoes,railing,glass, badhandv4, EasyNegative, ng_deepnegative_v1_75t",
            'ckpt': 'majicmixRealistic_v6.safetensors',
            'network_weights': ['train.safetensors'],
            'network_mul': [1.0],
            'scale': 7,
            'negative_scale': None,
            'seed': None,
            'sampler': 'euler-a',
            'steps': 25,
            'width': 512,
            'height': 768
        },
    )
    retain_code = ['200014']
    prompt_list = list(filter(lambda item: item['code'] in retain_code, prompt_list))
    return prompt_list


def transfer_prompt():
    import pandas as pd
    prompt_file = "/mnt/c/Users/Admin/Desktop/prompt.xlsx"
    prompt_pd = pd.read_excel(prompt_file)
    num = len(prompt_pd)
    prompt_list = []
    for i in range(num):
        network_weights = ['train.safetensors']
        network_mul = [1]
        lora_name = prompt_pd.iloc[i, 2]
        if not pd.isna(lora_name):
            if lora_name == 'polyhedron_skinny_all':
                network_weights.append(prompt_pd.iloc[i, 2] + '.pt')
            else:
                network_weights.append(prompt_pd.iloc[i, 2] + '.safetensors')
            network_mul.append(0.7)
        prompt_info = {
            'code': '200000',
            'posPrompt': prompt_pd.iloc[i, 0],
            'negPrompt': prompt_pd.iloc[i, 1],
            'ckpt': 'majicmixRealistic_v6.safetensors',
            'network_weights': network_weights,
            'network_mul': network_mul,
            'scale': 7,
            'negative_scale': None,
            'seed': None,
            'sampler': 'euler_a',
            'steps': 30,
            'width': 512,
            'height': 768
        }
        prompt_list.append(prompt_info)
    return prompt_list


def concat_images(image_path_list, valid_prompt_num, result_path, highres_fix=False, by_row=True):
    image_num = len(image_path_list)
    if image_num == 0:
        print('No image generated')
        return
    ROW = valid_prompt_num
    COL = image_num // ROW
    UNIT_WIDTH_SIZE = 512 if not highres_fix else 1024
    UNIT_HEIGHT_SIZE = 768 if not highres_fix else 1024
    image_files = []
    for index in range(COL*ROW):
        image_files.append(Image.open(image_path_list[index])) #读取所有用于拼接的图片
    target = Image.new('RGB', (UNIT_WIDTH_SIZE * COL, UNIT_HEIGHT_SIZE * ROW)) #创建成品图的画布
    #第一个参数RGB表示创建RGB彩色图，第二个参数传入元组指定图片大小，第三个参数可指定颜色，默认为黑色
    if by_row:
        for row in range(ROW):
            for col in range(COL):            
                #对图片进行逐行拼接
                #paste方法第一个参数指定需要拼接的图片，第二个参数为二元元组（指定复制位置的左上角坐标）
                #或四元元组（指定复制位置的左上角和右下角坐标）
                target.paste(image_files[COL*row + col], (0 + UNIT_WIDTH_SIZE*col, 0 + UNIT_HEIGHT_SIZE*row))
    else:
        for col in range(COL):
            for row in range(ROW):
                #对图片进行逐行拼接
                #paste方法第一个参数指定需要拼接的图片，第二个参数为二元元组（指定复制位置的左上角坐标）
                #或四元元组（指定复制位置的左上角和右下角坐标）
                target.paste(image_files[ROW*col + row], (0 + UNIT_WIDTH_SIZE*col, 0 + UNIT_HEIGHT_SIZE*row))
    target.save(result_path, quality=40) #成品图保存


def dict_to_image_name(params):
    name = ''
    for k, v in params.items():
        v = str(v)
        if '/' in v:
            v = v.split('/')[-1]
        name += str(k) + '=' + v + ' '
    return name + '.jpg'


if __name__ == '__main__':
    arg = sys.argv[2]
    run_train = True if arg == 'run_train' else False
    print('本次执行选项:', arg)

    raw_path = './raw_images'
    root_path = './train'
    train_image_name_list = ['zkj']
    train_image_sex_code_list = [100002]
    train_image_age_list = [25]
    params_dict_list = [
        {'base_model_path': 'majicmixRealistic_v6.safetensors', 'seed': 47},
        # {'base_model_path': 'dreamshaper_7.safetensors', 'seed': 47},
        # {'base_model_path': 'leosamsMoonfilm_filmGrain20.safetensors', 'seed': 47},
        # {'base_model_path': 'majicmixRealistic_v6.safetensors', 'seed': 47},
        # {'base_model_path': 'revAnimated_v122.safetensors', 'seed': 47},
    ]
    gen_params_dict = {'seed': 47}
    images_per_prompt = 4
    # style_res_list = transfer_prompt()
    style_res_list = generate_prompt()
    for name, sex_code, age in zip(train_image_name_list, train_image_sex_code_list, train_image_age_list):
        print(name)
        for params in params_dict_list:
            model_processor = ModelImageProcessor(user_id=None, 
                                                order_id=name, 
                                                sex_code=sex_code,
                                                age=age, 
                                                style_code='200001,200002,200004,200003,200006,200005,200007')
            image_recieve_path, image_crop_path = model_processor.prepare_paths(raw_path, num_repeat="5")
            preprocessor = ModelPreprocessing()
            if run_train:
                # copy_num = preprocessor.copy_image_from_path(image_recieve_path, image_crop_path)
                crop_num = preprocessor.crop_face_from_path_auto_scale(image_recieve_path, image_crop_path)
            # model_processor.generate_tags()
            valid_prompt_num, output_images = model_processor.process(style_res_list, images_per_prompt, run_train, gen_sample_image=False, use_step=-1, highres_fix=False, train_params=params, gen_params=gen_params_dict)
            concat_images(output_images, valid_prompt_num, os.path.join(root_path, name, dict_to_image_name(params)), highres_fix=False, by_row=True)
