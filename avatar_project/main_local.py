import os
import sys
import shutil
import dlib
import cv2
from PIL import Image
from train_network_online import train_online

class ModelPreprocessing:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def crop_face_from_path(self, input_path, crop_path) : 
        crop_num = 0 
        for root, dirs, files in os.walk(input_path):
            for file_name in files: 
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path    = os.path.join(root, file_name) 
                    cropped_image = self.extract_square_face_and_shoulders(image_path, scale=4)
                    cropped_face_image = self.extract_square_face_and_shoulders(image_path, scale=1.2) 
                    if cropped_image is not None : 
                        crop_num +=1 
                        cropped_image_save_path = os.path.join(crop_path, file_name) 
                        self.save_cropped_image(cropped_image_save_path,  cropped_image) 
                        cropped_face_image_save_path = os.path.join(crop_path, 'face_'+file_name) 
                        self.save_cropped_image(cropped_face_image_save_path,  cropped_face_image) 
                    else :
                        self.copy_image_to_folder(crop_path, image_path)   
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
    
    def copy_image_from_path(self, input_path, copy_path):
        for root, dirs, files in os.walk(input_path):
            for file_name in files: 
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path    = os.path.join(root, file_name)
                    self.copy_image_to_folder(copy_path, image_path)

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
        # with open(os.path.join(self.output_path, "prompt.txt"), 'w') as f:
        #     f.writelines(prompt_list)
        return valid_style_code_list
    
    def process(self, params={}):
        valid_style_code_list = self.generate_prompt()
        if len(valid_style_code_list) == 0:
            print('order_id:{},没有风格用于生成'.format(self.order_id))
            return []
        # 训练模型
        try:
            output_images = train_online(self.order_id, 
                                        self.model_input_path,
                                        self.model_path,
                                        self.log_path,
                                        self.output_path,
                                        **params)
        except Exception as e:
            print('order_id:{},执行出错 {}'.format(self.order_id, e))
            output_images = []
        return len(valid_style_code_list), output_images

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
            'neg': "tattooing,Neck decoration, collar, necklace,collar,badhandv4, paintings, sketches, (worst qualit:2), (low quality:2), (normal quality:2), lowers, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), manboobs, (backlight:1.2), double navel, muted arms, hused arms, neck lace, analog, analog effects, (sunglass:1.4), nipples, nsfw, bad architecture, watermark, (mole:1.5), EasyNegative"
        },
        '200002': {
            'pos': "mj3d style,3dmm,3d,(masterpiece, best quality:1.1), ghibli style, san \(mononoke hime\), 1{sex}, armlet, bangs, black hair, black undershirt, breasts, cape, circlet, earrings, facepaint, floating hair, forest, fur cape, green eyes, jewelry, looking at viewer, medium breasts, nature, necklace, outdoors, parted bangs, shirt, short hair, sleeveless, sleeveless shirt, solo, tooth necklace, tree, upper body, white shirt, {age}".format(sex=sex_str, age=age_str),
            'neg': "badhandv4, paintings, sketches, (worst qualit:2), (low quality:2), (normal quality:2), lowers, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), manboobs, (backlight:1.2), double navel, muted arms, hused arms, neck lace, analog, analog effects, (sunglass:1.4), nipples, nsfw, bad architecture, watermark, (mole:1.5), EasyNegative"
        },
        '200003': {
            'pos': "8k portrait of beautiful cyborg with brown hair, intricate, elegant, highly detailed, majestic, digital photography, art by artgerm and ruan jia and greg rutkowski surreal painting gold butterfly filigree, broken glass, (masterpiece, sidelighting, finely detailed beautiful eyes: 1.2), hdr, 1{sex}, {age}".format(sex=sex_str, age=age_str),
            'neg': ""
        },
        '200004': {
            'pos': "{sex} holding cat, cat ears, chibi, blue, gold, white, purpple, dragon scaly armor, forest background, fantasy style, (dark shot:1.17), epic realistic, faded, ((neutral colors)), art, (hdr:1.5), (muted colors:1.2), hyperdetailed, (artstation:1.5), cinematic, warm lights, dramatic light, (intricate details:1.1), complex background, (rutkowski:0.8), (teal and orange:0.4), colorfull, (natural skin texture, hyperrealism, soft light, sharp:1.2), (intricate details:1.12), hdr, (intricate details, hyperdetailed:1.15), white hair, {age}".format(sex=sex_str, age=age_str),
            'neg': ""
        },
        '200005': {
            'pos': "Ambilight, masterpiece, ultra-high quality,( ultra detailed original illustration),( 1{sex}, upper body),(( harajuku fashion)),(( flowers with human eyes, flower eyes)), double exposure, fussion of fluid abstract art, glitch,( 2d),( original illustration composition),( fusion of limited color, maximalism artstyle, geometric artstyle, butterflies, junk art), {age}".format(sex=sex_str, age=age_str),
            'neg': "easyNegative,(realistic),(3d face),(worst quality:1.2), (low quality:1.2), (lowres:1.1), (monochrome:1.1), (greyscale),(multiple legs:1.5),(extra legs:1.5),(wrong legs),(multiple hands),(missing limb),(multiple bodies:1.5),garter straps,multiple heels,legwear,thghhighs,stockings,golden shoes,railing,glass"
        },
        '200006': {
            'pos': "masterpiece, best quality,realistic,(realskin:1.5),1{sex},school,longhair,no_bangs, side_view,looking at viewer,school uniform,realskin softlight, {age}".format(sex=sex_str, age=age_str),
            'neg': "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, ((watermark:2)), (white letters:1), (multi nipples), bad anatomy, bad hands, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, worst quality, jpeg artifacts, signature, watermark, username, bad feet, Multiple people, blurry, poorly drawn hands, poorly drawn face, mutation, deformed, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed, mutated hands, polar lowres, bad body, bad proportions, gross proportions, wrong feet bottom render, abdominal stretch, briefs, knickers, kecks, thong, fused fingers, bad body,bad proportion body to legs, wrong toes, extra toes, missing toes, weird toes, 2 body, 2 pussy, 2 upper, 2 lower, 2 head, 3 hand, 3 feet, extra long leg, super long leg, mirrored image, mirrored noise,, badhandv4, ng_deepnegative_v1_75t"
        },
        # '200007': {
        #     'pos': "((master piece)),best quality, illustration, dark, 1{sex}, In the wilderness,High mountain,Snow-capped mountains in the distance, castle, beautiful detailed eyes, beautiful detailed hair, {age}".format(sex=sex_str, age=age_str),
        #     'neg': "sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((grayscale)), skin spots, skin blemishes, bad anatomy, ((monochrome)), (((extra legs))), ((grayscale)),DeepNegative, tilted head, lowres, bad a natomy, bad hands, text, error, fewer digits, cropped, worstquality, low quality, bad legs, fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,missing fingers,missing arms,missing legs,extra digit , extra arms, extra leg, extra foot"
        # },
        # '200008': {
        #     'pos': "(masterpiece, best quality:1.2), from side, solo, {sex2} focus, 1{sex}, aomine daiki, muscular, serious, closed mouth, sportswear, basketball uniform, basketball court, {age}".format(sex=sex_str, sex2=sex2_str, age=age_str),
        #     'neg': ""
        # },
    }
    return prompt_dict


def concat_images(image_path_list, valid_prompt_num, result_path):
    image_num = len(image_path_list)
    ROW = valid_prompt_num
    COL = image_num // ROW
    UNIT_WIDTH_SIZE = 512
    UNIT_HEIGHT_SIZE = 512
    image_files = []
    for index in range(COL*ROW):
        image_files.append(Image.open(image_path_list[index])) #读取所有用于拼接的图片
    target = Image.new('RGB', (UNIT_WIDTH_SIZE * COL, UNIT_HEIGHT_SIZE * ROW)) #创建成品图的画布
    #第一个参数RGB表示创建RGB彩色图，第二个参数传入元组指定图片大小，第三个参数可指定颜色，默认为黑色
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
    raw_path = './raw_images'
    root_path = './train'
    train_image_name_list = ['zkj4', ]
    train_image_sex_code_list = [100002, ]
    train_image_age_list = [30, ]
    params_dict_list = [
        {'base_model_path': './models/stable-diffusion/chilloutmix_NiPrunedFp32Fix.safetensors', 'text_encoder_lr': 2e-5,
                'unet_lr':2e-5, 'learning_rate':2e-5, 'seed': 47},
        # {'base_model_path': './models/stable-diffusion/dreamshaper_631BakedVae.safetensors', 'seed': 47},
    ]
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
            image_recieve_path, image_crop_path = model_processor.prepare_paths(raw_path)
            preprocessor = ModelPreprocessing()
            crop_num = preprocessor.crop_face_from_path(image_recieve_path, image_crop_path)
            # crop_num = preprocessor.copy_image_from_path(image_recieve_path, image_crop_path)
            valid_prompt_num, output_images = model_processor.process(params)
            concat_images(output_images, valid_prompt_num, os.path.join(root_path, name, dict_to_image_name(params)))
