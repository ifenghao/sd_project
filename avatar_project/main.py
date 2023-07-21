import os
import json
import time
from mysql_manager import MysqlManager
from oss_manager import HandleOSSUtil   
from logger_manager import init_logger
from collections import defaultdict 
from model_train_predict import ModelImageProcessor
from model_preprocessing import ModelPreprocessing
from urllib.parse import urlparse,urljoin
import requests 
import hashlib
from jinja2 import Template
import traceback

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")  
logger  = init_logger(log_dir, "main") 
config  = json.load(open("conf.json", encoding='utf-8'))
server_code = config['server_code']
api_key     = config['api_config']['api_key']
api_url_base= config['api_config']['api_url_base'] 

handle_oss_util = HandleOSSUtil(key_id=config["oss_config"]["key_id"], 
                                key_secret=config["oss_config"]["key_secret"], 
                                bucket=config["oss_config"]["bucket_name"])
mysql_manager_conn = MysqlManager() #创建连接池

 
def add_md5(serverCode,timeStamp,apiKey=api_key) :
    data     = serverCode + timeStamp + apiKey 
    md5_hash = hashlib.md5()
    md5_hash.update(data.encode())
    md5_digest = md5_hash.hexdigest()
    return md5_digest 

def fetch_order_data(apiUrlBase=api_url_base, serverCode=str(server_code)): 
    endpoint="/god/order/task" 
    url = urljoin(api_url_base, endpoint) 
    timeStamp = str(int(time.time())) 
    sign = add_md5(serverCode,timeStamp)
    
    data = {'serverCode':serverCode,'sign':sign, 'timeStamp':timeStamp} 
    headers = {'content-type': 'application/json'} 
    try:
        res = requests.post(url, data=json.dumps(data), headers=headers) 
        return json.loads(res.text)
    except Exception as e:
        logger.error('取订单接口出错{}'.format(e))
        return None 
    
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

def convert_style_res_list(style_res_list, gender_des, gender, age_des):
    style_infos = [] 
    for style_res in style_res_list : 
        code = style_res.get('code','')
        gen_params_dict = {}
        pos_prompt = style_res.get('posPrompt','')
        neg_prompt = style_res.get('negPrompt','')
        mod_pos_prompt = modified_prompt(pos_prompt,gender_des,gender,age_des) 
        mod_neg_prompt = modified_prompt(neg_prompt,gender_des,gender,age_des) 
        prompt = "{} --n {}".format(mod_pos_prompt, mod_neg_prompt)
        network_weights = style_res.get('network_weights',['train.safetensors'])  #默认值是['train.safetensors']
        network_mul     = style_res.get('network_mul',[1.0])                      #默认值是[1.0]
        ckpt    = style_res.get('ckpt','dreamshaper_631BakedVae.safetensors') 
        sampler = style_res.get('sampler','euler') 
        steps   = style_res.get('steps',30)   
        width   = style_res.get('width',512)   
        height  = style_res.get('height',512) 
        gen_params_dict['code']   = code 
        gen_params_dict['prompt'] = prompt 
        gen_params_dict['ckpt']   = "./models/stable-diffusion/"+ckpt
        gen_params_dict['network_weights'] = [network_weights[0]] + ['./models/lora/' + i for i in network_weights[1:]]
        gen_params_dict['network_mul'] = network_mul
        gen_params_dict['sampler'] = sampler 
        gen_params_dict['steps'] = steps
        gen_params_dict['W'] = width 
        gen_params_dict['H'] = height 
        style_infos.append(gen_params_dict)
    return style_infos

def insert_ai_order_photo(user_id, order_id, output_dict):
    ''':Description:结果图片url 存入 mm_ai_order_photo,
    改成一个风格进行批量插入'''
    affected_rows_all = 0
    try:
        insert_sql = "INSERT INTO mm_ai_order_photo (user_id, order_id, server_code, style_code, photo_url) VALUES (%s, %s,%s,%s, %s)"
        for style_code, photo_urls in output_dict.items():
            values = []
            for photo_url in photo_urls:
                values.append((user_id, order_id, server_code, style_code, photo_url))      
            if values:
                affected_rows = mysql_manager_conn.insertMany(insert_sql, values)
                affected_rows_all +=affected_rows
    
    except Exception as e:
        logger.error('order_id:{},insert_ai_order_photo出错{}'.format(order_id, e))
    return affected_rows_all

def update_order_status_res(res_status,order_id):
    affected_rows = 0
    try:
        update_sql = "UPDATE mm_order SET order_status = %s WHERE order_id = %s AND order_status = 2"
        affected_rows = mysql_manager_conn.update(update_sql, (res_status,order_id,))    
    except Exception as e:
        logger.error('order_id:{},update_order_status_res出错{}'.format(order_id, e))
    return affected_rows


if __name__ == '__main__':
    while True:
        # STEP1: 使用接口抢单
        fetch_result   = fetch_order_data()
        if fetch_result is not None and fetch_result['code']=='200': 
            fetch_data = fetch_result['data']
            order_id   = fetch_data.get("orderId")  
            user_id    = fetch_data.get("userId")  
            style_code = fetch_data.get("styleCode") 
            logger.info('抢单成功,user_id:{},order_id:{},style_code:{}'.format(user_id,order_id,style_code))
            try: 
                sex_code   = fetch_data.get("sexCode")
                age = fetch_data.get("age") 
                photos_info    = fetch_data.get('photoUrls',[]) 
                style_res_list = fetch_data.get("styleResList",[])  
                gender_des, gender, age_des = preprocess_gender_age(sex_code,age,reverse_gender=False)   
                model_params = {"order_id":order_id}  
                style_infos  = convert_style_res_list(style_res_list, gender_des, gender, age_des)
                model_params['style_infos'] = style_infos
                logger.info('风格预处理后:{}'.format(model_params))
            
                #STEP2: 提取用户上传图片,下载到input文件夹
                if photos_info and len(photos_info)>0 : 
                    model_processor = ModelImageProcessor(logger, user_id, order_id, sex_code, age, style_code)  
                    # TODO 改这里
#                     model_processor = ModelImageProcessor(logger, order_id, model_params) 
                    image_recieve_path, image_crop_path = model_processor.prepare_paths()

                    num_photo = 0 
                    for oss_input_file in photos_info : 
                        downloadoss_result= handle_oss_util.download_one_file(oss_input_file, image_recieve_path)
                        if downloadoss_result ==1:
                            num_photo+=1
                    logger.info('order_id:{},共下载{}张图片'.format(order_id,str(num_photo)))

                    #STEP 2.5: 对下载的图片进行截取
                    preprocessor = ModelPreprocessing(logger)
                    # copy_num = preprocessor.copy_image_from_path(image_recieve_path, image_crop_path)
                    copy_num, crop_num = preprocessor.crop_face_from_path_auto_scale(image_recieve_path, image_crop_path)
                    logger.info('order_id:{},共复制{}张/截取{}张图片'.format(order_id,str(copy_num),str(crop_num)))
                        
                    #STEP3 train model and predict
                    local_output_dict = model_processor.process_with_gen(gen_sample_image=False, use_step=-1, highres_fix=True)
                    print(local_output_dict)
                    logger.info('order_id:{},模型训练和预测结束'.format(order_id)) 
                    
                    # STEP4: 上传oss 并获取url  
                    num_upload_photo = 0
                    oss_output_dict  = defaultdict(list)
                    oss_dir = config["oss_config"]["output_file"] 
                    for style,local_output in local_output_dict.items():
                        for file_path in local_output:
                            file_name = file_path.split('/')[-1]
                            oss_path  = f'{oss_dir}/{file_name}' 
                            try: 
                                uploadoss_result = handle_oss_util.update_one_file(oss_path,file_path) 
                                if uploadoss_result :
                                    num_upload_photo +=1 
                                    oss_output_dict[style].append(oss_path) 
                            except Exception as e:
                                logger.error('上传文件时发生异常:{},order_id:{},file_name:{}'.format(str(e),order_id,file_name))
                    logger.info('order_id:{},结果图片上传到oss成功, 共{}个风格,共存入{}张'.format(order_id,len(local_output_dict),num_upload_photo)) 
                
                    # STEP5: 将结果url 存入 mm_ai_order_photo
                    num_insert_photo = insert_ai_order_photo(user_id, order_id, oss_output_dict)
                    if num_insert_photo > 0 :
                        logger.info('order_id:{},结果insert 成功,共insert{}张'.format(order_id,num_insert_photo)) 
                    else:
                        logger.error('order_id:{},结果insert 失败'.format(order_id)) 
                    
                    # STEP6: 更新mm_order 中的状态
                    num_update_status3 = update_order_status_res(3,order_id) 
                    if num_update_status3 == 1 :
                        logger.info('order_id:{},update_status3 成功'.format(order_id)) 
                    else: 
                        logger.error('order_id:{},update_status3 失败'.format(order_id)) 
            except Exception as e:
                num_update_status5 = update_order_status_res(5,order_id) 
                if num_update_status5 != 1 :
                    logger.error('order_id:{},update_status5 失败'.format(order_id)) 
                logger.error("抢单后发生异常,update5,异常回溯:{},order_id:{}".format(traceback.format_exc(),order_id) )
        else:
            logger.warning('本次未抢到单')
        time.sleep(5)  # 休眠5秒 
