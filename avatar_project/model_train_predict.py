import os
import json
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

    def prepare_paths(self, num_repeat="30"):
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

    def generate_prompt(self):
        sex_str, sex2_str, age_str = generate_info_prompt(self.sex_code, self.age)
        prompt_dict = generate_prompt_dict(sex_str, sex2_str, age_str)
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
    
    def gen_img(self, use_step=-1, params={}):
        model_file_list = os.listdir(self.model_path)
        model_file_list = list(filter(lambda t: t.startswith(self.order_id), model_file_list))
        total_steps = len(model_file_list)
        if total_steps == 0 or use_step >= total_steps:
            self.logger.error('order_id:{},No lora model is loaded'.format(self.order_id))
            return []
        model_file = model_file_list[use_step]
        self.logger.error('order_id:{},select model file: {}'.format(self.order_id, model_file))
        # 生成图片
        try:
            output_gen_images = gen_img(outdir=self.output_path, 
                                        network_weights=os.path.join(self.model_path, model_file), 
                                        from_file=os.path.join(self.output_path, 'prompt.txt'),
                                        **params)
        except Exception as e:
            self.logger.error('order_id:{},生成出错 {}'.format(self.order_id, e))
            output_gen_images = []
        return output_gen_images

    def process_with_gen(self, gen_sample_image=True, use_step=-1, train_params={}, gen_params={}):
        valid_style_code_list = self.generate_prompt()
        if len(valid_style_code_list) == 0:
            self.logger.info('order_id:{},没有风格用于生成'.format(self.order_id))
            return {}
        output_sample_images = self.train(gen_sample_image, params=train_params)
        output_gen_images = self.gen_img(use_step, params=gen_params)
        output_images = output_gen_images
        # 整理图片结果
        num_style = len(valid_style_code_list)
        num_image_per_style = len(output_images) // num_style
        style_image_list = []
        for i in range(num_style):
            style_image_list.append([])
        for i, image in enumerate(output_images):
            style_image_list[i // num_image_per_style].append(image)
        # 返回图片地址字典
        image_dict = dict(zip(valid_style_code_list, style_image_list))
        return image_dict
    
    def process(self):
        valid_style_code_list = self.generate_prompt()
        if len(valid_style_code_list) == 0:
            self.logger.info('order_id:{},没有风格用于生成'.format(self.order_id))
            return []
        # 训练模型
        try:
            output_images = train_online(self.order_id, 
                                        self.model_input_path,
                                        self.model_path,
                                        self.log_path,
                                        self.output_path)
        except Exception as e:
            self.logger.error('order_id:{},执行出错 {}'.format(self.order_id, e))
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
    sex2_str = 'male'
    age_str = '{age} year old'.format(age=age)
    male_obj_list = ['boy', ' young man', 'man']
    male_age_list = [16, 28]
    female_obj_list = ['girl', ' young woman', 'woman']
    female_age_list = [26, 40]

    if int(sex_code) == 100001:
        sex_str = male_obj_list[get_age_index(age, male_age_list)]
        sex2_str = 'male'
    elif int(sex_code) == 100002:
        sex_str = female_obj_list[get_age_index(age, female_age_list)]
        sex2_str = 'female'
    return sex_str, sex2_str, age_str


def generate_prompt_dict(sex_str, sex2_str, age_str):
    prompt_dict = {
        '200001': {
            'pos': "3dmm style,(masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2), (fractal art:1.3), 1{sex}, beautiful, high detailed, purple hair with a hint of pink, pink eyes, dark lighting, serious face, looking the sky, sky, medium shot, black sweater, jewelry, {age}".format(sex=sex_str, age=age_str),
            'neg': "tattooing,Neck decoration, collar, necklace,collar,badhandv4, paintings, sketches, (worst qualit:2), (low quality:2), (normal quality:2), lowers, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), manboobs, (backlight:1.2), double navel, muted arms, hused arms, neck lace, analog, analog effects, (sunglass:1.4), nipples, nsfw, bad architecture, watermark, (mole:1.5), EasyNegative, ng_deepnegative_v1_75t"
        },
        '200002': {
            'pos': "mj3d style,3dmm,3d,(masterpiece, best quality:1.1), elf, light blue hair, glasses, mole on mouth ,anime , (smile:0.5), 1{sex}, {age}".format(sex=sex_str, age=age_str),
            'neg': "badhandv4, paintings, sketches, (worst qualit:2), (low quality:2), (normal quality:2), lowers, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), manboobs, (backlight:1.2), double navel, muted arms, hused arms, neck lace, analog, analog effects, (sunglass:1.4), nipples, nsfw, bad architecture, watermark, (mole:1.5), EasyNegative, ng_deepnegative_v1_75t"
        },
        '200003': {
            'pos': "8k portrait of beautiful cyborg with brown hair, intricate, elegant, highly detailed, majestic, digital photography, art by artgerm and ruan jia and greg rutkowski surreal painting gold butterfly filigree, broken glass, (masterpiece, sidelighting, finely detailed beautiful eyes: 1.2), hdr, 1{sex}, {age}".format(sex=sex_str, age=age_str),
            'neg': "sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, ((2{sex})), (deformed fingers:1.2), (long fingers:1.2),succubus wings,horn,succubus horn,succubus hairstyle, (bad-artist-anime), bad-artist, bad hand, badhandv4, EasyNegative, ng_deepnegative_v1_75t".format(sex=sex_str)
        },
        '200004': {
            'pos': "((master piece)),best quality, illustration, 1{sex}, Look out the window, beautiful detailed eyes, (beautiful detailed cyberpunk city), beautiful detailed hair, {age}".format(sex=sex_str, age=age_str),
            'neg': "sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((grayscale)), skin spots, skin blemishes, bad anatomy, ((monochrome)), (((extra legs))), ((grayscale)),DeepNegative, tilted head, lowres, bad a natomy, bad hands, text, error, fewer digits, cropped, worstquality, low quality, bad legs, fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,missing fingers,missing arms,missing legs,extra digit , extra arms, extra leg, extra foot, badhandv4, EasyNegative, ng_deepnegative_v1_75t"
        },
        '200005': {
            'pos': "Ambilight, masterpiece, ultra-high quality,( ultra detailed original illustration),( 1{sex}, upper body),(( harajuku fashion)),(( flowers with human eyes, flower eyes)), double exposure, fussion of fluid abstract art, glitch,( 2d),( original illustration composition),( fusion of limited color, maximalism artstyle, geometric artstyle, butterflies, junk art), {age}".format(sex=sex_str, age=age_str),
            'neg': "easyNegative,(realistic),(3d face),(worst quality:1.2), (low quality:1.2), (lowres:1.1), (monochrome:1.1), (greyscale),(multiple legs:1.5),(extra legs:1.5),(wrong legs),(multiple hands),(missing limb),(multiple bodies:1.5),garter straps,multiple heels,legwear,thghhighs,stockings,golden shoes,railing,glass, badhandv4, EasyNegative, ng_deepnegative_v1_75t"
        },
        '200006': {
            'pos': "1{sex}, hand_gesture, white_hair, multicolored hair, long hair, very long hair, multicolored eyes, (multicolored_background:1.8), solo, smile, looking at viewer, cherry blossoms, grin, hair between eyes, dress, dress, dress_shirt, white_dress, multicolored_dress, bangs, upper body, album cover, {age}".format(sex=sex_str, age=age_str),
            'neg': "sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, ((2{sex})), (deformed fingers:1.2), (long fingers:1.2),succubus wings,horn,succubus horn,succubus hairstyle, (bad-artist-anime), bad-artist, bad hand, badhandv4, EasyNegative, ng_deepnegative_v1_75t".format(sex=sex_str)
        },
        '200007': {
            'pos': "((master piece)),best quality, illustration, dark, 1{sex}, In the wilderness,High mountain,Snow-capped mountains in the distance, castle, beautiful detailed eyes, beautiful detailed hair, {age}".format(sex=sex_str, age=age_str),
            'neg': "sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((grayscale)), skin spots, skin blemishes, bad anatomy, ((monochrome)), (((extra legs))), ((grayscale)),DeepNegative, tilted head, lowres, bad a natomy, bad hands, text, error, fewer digits, cropped, worstquality, low quality, bad legs, fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,missing fingers,missing arms,missing legs,extra digit , extra arms, extra leg, extra foot, badhandv4, EasyNegative, ng_deepnegative_v1_75t"
        },
        # '200008': {
        #     'pos': "(masterpiece, best quality:1.2), from side, solo, {sex2} focus, 1{sex}, aomine daiki, muscular, serious, closed mouth, sportswear, basketball uniform, basketball court, {age}".format(sex=sex_str, sex2=sex2_str, age=age_str),
        #     'neg': ""
        # },
    }
    return prompt_dict
