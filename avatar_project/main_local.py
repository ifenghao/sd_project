import os
import sys
import shutil
import dlib
import cv2
from PIL import Image
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
        
    def get_face_detect_result(self, image_path):
        try : 
            image = cv2.imread(image_path) 
            gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) != 1 : 
                return None, None 
            else :  
                return image, faces[0]
        except Exception as e:
            print(f"截取头像失败:{e},{image_path}")  
            return None, None
        
    def get_max_face_scale(self, image, face):
        left, top, w, h = face.left(), face.top(), face.width(), face.height()
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
        left, top, w, h = face.left(), face.top(), face.width(), face.height()
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

    def prepare_paths(self, raw_path, num_repeat="50"):
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
            print('order_id:{},训练出错 {}'.format(self.order_id, e))
            output_sample_images = []
        return output_sample_images
    
    def generate(self, use_step=-1, highres_fix=False, params={}):
        model_file_list = os.listdir(self.model_path)
        model_file_list = list(filter(lambda t: t.startswith(self.order_id), model_file_list))
        total_steps = len(model_file_list)
        if total_steps == 0 or use_step >= total_steps:
            print('No lora model is loaded')
            return []
        model_file = model_file_list[use_step]
        print('select model file: ' + model_file)
        # 生成图片
        try:
            output_gen_images = gen_img(outdir=self.output_path, 
                                        network_weights=os.path.join(self.model_path, model_file), 
                                        from_file=os.path.join(self.output_path, 'prompt.txt'),
                                        highres_fix=highres_fix,
                                        **params)
        except Exception as e:
            print('order_id:{},生成出错 {}'.format(self.order_id, e))
            output_gen_images = []
        return output_gen_images
    
    def process(self, run_train=True, gen_sample_image=True, use_step=-1, highres_fix=False, train_params={}, gen_params={}):
        valid_style_code_list = self.generate_prompt()
        if len(valid_style_code_list) == 0:
            print('order_id:{},没有风格用于生成'.format(self.order_id))
            return []
        if run_train:
            output_sample_images = self.train(gen_sample_image, params=train_params)
        output_gen_images = self.generate(use_step, highres_fix, params=gen_params)
        return len(valid_style_code_list), output_gen_images

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
            'pos': "3dmm style,(masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2), (fractal art:1.3), 1{sex}, upper body, beautiful, high detailed, purple hair with a hint of pink, pink eyes, dark lighting, serious face, looking the sky, sky, medium shot, black sweater, jewelry, {age}".format(sex=sex_str, age=age_str),
            'neg': "tattooing,Neck decoration, collar, necklace,collar,badhandv4, paintings, sketches, (worst qualit:2), (low quality:2), (normal quality:2), lowers, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), manboobs, (backlight:1.2), double navel, muted arms, hused arms, neck lace, analog, analog effects, (sunglass:1.4), nipples, nsfw, bad architecture, watermark, (mole:1.5), EasyNegative, ng_deepnegative_v1_75t"
        },
        '200002': {
            'pos': "mj3d style,3dmm,3d,(masterpiece, best quality:1.1), elf, light blue hair, glasses, mole on mouth ,anime , (smile:0.5), 1{sex}, upper body, {age}".format(sex=sex_str, age=age_str),
            'neg': "badhandv4, paintings, sketches, (worst qualit:2), (low quality:2), (normal quality:2), lowers, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), manboobs, (backlight:1.2), double navel, muted arms, hused arms, neck lace, analog, analog effects, (sunglass:1.4), nipples, nsfw, bad architecture, watermark, (mole:1.5), EasyNegative, ng_deepnegative_v1_75t"
        },
        '200003': {
            'pos': "8k portrait of beautiful cyborg with brown hair, intricate, elegant, highly detailed, majestic, digital photography, art by artgerm and ruan jia and greg rutkowski surreal painting gold butterfly filigree, broken glass, (masterpiece, sidelighting, finely detailed beautiful eyes: 1.2), hdr, 1{sex}, upper body, {age}".format(sex=sex_str, age=age_str),
            'neg': "sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, ((2{sex})), (deformed fingers:1.2), (long fingers:1.2),succubus wings,horn,succubus horn,succubus hairstyle, (bad-artist-anime), bad-artist, bad hand, badhandv4, EasyNegative, ng_deepnegative_v1_75t".format(sex=sex_str)
        },
        '200004': {
            'pos': "((master piece)),best quality, illustration, 1{sex}, upper body, Look out the window, beautiful detailed eyes, (beautiful detailed cyberpunk city), beautiful detailed hair, {age}".format(sex=sex_str, age=age_str),
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
            'pos': "((master piece)),best quality, illustration, dark, 1{sex}, upper body, In the wilderness,High mountain,Snow-capped mountains in the distance, castle, beautiful detailed eyes, beautiful detailed hair, {age}".format(sex=sex_str, age=age_str),
            'neg': "sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((grayscale)), skin spots, skin blemishes, bad anatomy, ((monochrome)), (((extra legs))), ((grayscale)),DeepNegative, tilted head, lowres, bad a natomy, bad hands, text, error, fewer digits, cropped, worstquality, low quality, bad legs, fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,missing fingers,missing arms,missing legs,extra digit , extra arms, extra leg, extra foot, badhandv4, EasyNegative, ng_deepnegative_v1_75t"
        },
        # '200008': {
        #     'pos': "(masterpiece, best quality:1.2), from side, solo, {sex2} focus, 1{sex}, aomine daiki, muscular, serious, closed mouth, sportswear, basketball uniform, basketball court, {age}".format(sex=sex_str, sex2=sex2_str, age=age_str),
        #     'neg': ""
        # },
    }
    return prompt_dict


def concat_images(image_path_list, valid_prompt_num, result_path, highres_fix=False, by_row=True):
    image_num = len(image_path_list)
    if image_num == 0:
        print('No image generated')
        return
    ROW = valid_prompt_num
    COL = image_num // ROW
    UNIT_WIDTH_SIZE = 512 if not highres_fix else 1024
    UNIT_HEIGHT_SIZE = 512 if not highres_fix else 1024
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
    train_image_name_list = ['zfh', ]
    train_image_sex_code_list = [100001, ]
    train_image_age_list = [26, ]
    params_dict_list = [
        {'text_encoder_lr': 2e-5, 'unet_lr':2e-5, 'learning_rate':2e-5, 'seed': 47},
        # {'base_model_path': './models/stable-diffusion/dreamshaper_631BakedVae.safetensors', 'seed': 47},
    ]
    gen_params_dict = {'images_per_prompt': 2, 'network_mul': 1, 'steps': 30, 'sampler': 'euler_a', 'seed': None}
    name_list = os.listdir(raw_path)
    name_list = list(filter(lambda t: os.path.isdir(os.path.join(raw_path, t)) and t in train_image_name_list, sorted(name_list)))
    print(name_list)
    for name, sex_code, age in zip(name_list, train_image_sex_code_list, train_image_age_list):
        for params in params_dict_list:
            model_processor = ModelImageProcessor(user_id=None, 
                                                order_id=name, 
                                                sex_code=sex_code,
                                                age=age, 
                                                style_code='200001,200002,200004,200003,200006,200005,200007')
            image_recieve_path, image_crop_path = model_processor.prepare_paths(raw_path, num_repeat="20")
            preprocessor = ModelPreprocessing()
            # copy_num = preprocessor.copy_image_from_path(image_recieve_path, image_crop_path)
            crop_num = preprocessor.crop_face_from_path_auto_scale(image_recieve_path, image_crop_path)
            # model_processor.generate_tags()
            valid_prompt_num, output_images = model_processor.process(run_train, gen_sample_image=False, use_step=-1, highres_fix=True, train_params=params, gen_params=gen_params_dict)
            concat_images(output_images, valid_prompt_num, os.path.join(root_path, name, dict_to_image_name(params)), highres_fix=True, by_row=True)
