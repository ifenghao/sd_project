import os
import json
import time
from mysql_manager import MysqlManager
from oss_manager import HandleOSSUtil   
from logger_manager import init_logger
from collections import defaultdict 
from model_train_predict import ModelImageProcessor
from urllib.parse import urlparse

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")  
# log_dir = "logs" 
logger  = init_logger(log_dir, "main") 
config  = json.load(open("conf.json", encoding='utf-8'))
server_code = '0' #temp_add, 机器号
handle_oss_util = HandleOSSUtil(key_id=config["oss_config"]["key_id"], 
                                key_secret=config["oss_config"]["key_secret"], 
                                bucket=config["oss_config"]["bucket_name"]) 
detector = dlib.get_frontal_face_detector() 

def extract_face_and_shoulders(image_path, scale_x=2.8, scale_y=3):
    try : 
        image = cv2.imread(image_path) 
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

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
        logger.error(f"截取头像失败:{e},{image_path}")  
        return None 

def copy_image_to_folder(destination_folder, source_file):
    try:        
        shutil.copy2(source_file, destination_folder)
    except Exception as e:
        logger.error(f"Error copying file:{e}") 

def save_cropped_image(cropped_image_path, cropped_image): 
    try:
        cv2.imwrite(cropped_image_path, cropped_image)
    except Exception as e:
        logger.error(f"Error saving cropped image: {e}")     

def crop_face_from_path(local_input_path,local_input_crop_path) : 
    crop_num = 0 
    for root, dirs, files in os.walk(local_input_path):
        for file_name in files: 
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path    = os.path.join(root, file_name) 
                cropped_image = extract_face_and_shoulders(image_path, scale_x=2.8, scale_y=3)
                if cropped_image is not None : 
                    crop_num +=1 
                    cropped_image_save_path = os.path.join(local_input_crop_path, file_name) 
                    save_cropped_image(cropped_image_save_path,  cropped_image) 
                else :
                    copy_image_to_folder(local_input_crop_path,image_path)   
    return crop_num

def grab_order(): 
    ''':Description:抢单'''
    mysql_manager_conn = MysqlManager() 
    select_sql = "SELECT * FROM mm_order WHERE order_status = 1 AND is_deleted = 0 ORDER BY pay_time ASC LIMIT 1"
    result     = mysql_manager_conn.getOne(select_sql)

    if result:
        order_id   = result['order_id']
        update_sql = "UPDATE mm_order SET order_status = 2,server_code =%s WHERE order_id = %s AND order_status = 1"
        affected_rows = mysql_manager_conn.update(update_sql, (server_code,order_id,))
        mysql_manager_conn.dispose()

        if affected_rows == 1:
            return result 
    mysql_manager_conn.dispose()
    return None

def get_order_photos(order_id):
    ''':Description:根据order_id 从mm_order_photo获取用户上传的照片'''
    mysql_manager_conn = MysqlManager()
    photo_sql   = "SELECT * FROM mm_order_photo WHERE order_id = %s"
    photos_info = mysql_manager_conn.getAll(photo_sql, (order_id,)) #多条数据
    mysql_manager_conn.dispose()
    return photos_info

def insert_ai_order_photo(user_id, order_id, output_dict):
    ''':Description:结果图片url 存入 mm_ai_order_photo'''
    mysql_manager_conn = MysqlManager()
    affected_rows = 0
    insert_sql = "INSERT INTO mm_ai_order_photo (user_id, order_id, server_code, style_code, photo_url) VALUES (%s, %s,%s,%s, %s)"
    values = []
    for style_code, photo_urls in output_dict.items():
        for photo_url in photo_urls:
            values.append((user_id, order_id, server_code, style_code, photo_url))      
    if values:
        affected_rows = mysql_manager_conn.insertMany(insert_sql, values)
    
    mysql_manager_conn.dispose()
    return affected_rows

def update_order_status(order_id):
    mysql_manager_conn = MysqlManager()

    update_sql = "UPDATE mm_order SET order_status = 3 WHERE order_id = %s AND order_status = 2"
    affected_rows = mysql_manager_conn.update(update_sql, (order_id,))
    
    mysql_manager_conn.dispose()
    return affected_rows


if __name__ == '__main__':
    while True:
        # STEP1: 抢单
        grab_result = grab_order() 
        if grab_result:
            # 提取用户信息
            user_id    = grab_result['user_id']
            order_id   = grab_result['order_id']
            sex_code   = grab_result['sex_code']
            age        = grab_result['age']
            style_code = grab_result['style_code']
            logger.info('抢单成功,user_id:{},order_id:{},sex_code:{},age:{},style_code:{}'.format(user_id,order_id,sex_code,age,style_code))
            
            #STEP2: 提取用户上传图片,下载到input文件夹
            photos_info = get_order_photos(order_id)
            if photos_info and len(photos_info)>0 : 
                processor = ModelImageProcessor(user_id, order_id, sex_code, age, style_code) 
                local_input_path, local_input_crop_path = processor.prepare_paths()

                num_photo = 0 
                for photo_info in photos_info : 
                    oss_input_file    = photo_info['photo_url']  
                    downloadoss_result= handle_oss_util.download_one_file(oss_input_file, local_input_path)
                    if downloadoss_result ==1:
                        num_photo+=1
                logger.info('order_id:{},共下载{}张图片'.format(order_id,str(num_photo)))

                #STEP 2.5: 对下载的图片进行截取
                crop_num = crop_face_from_path(local_input_path,local_input_crop_path)
                logger.info('order_id:{},共截取{}张图片'.format(order_id,str(crop_num)))
                    
                #TODO1: STEP3 train model and predict
                local_output_dict = processor.process(logger)
                print(local_output_dict)
                logger.info('order_id:{},模型训练和预测结束'.format(order_id)) 
                
                if len(local_output_dict) > 0 : 
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
                    logger.info('order_id:{},共上传{}张图片到oss'.format(order_id,str(num_upload_photo))) 
                  
                    # STEP5: 将结果url 存入 mm_ai_order_photo
                    num_insert_photo = insert_ai_order_photo(user_id, order_id, oss_output_dict)
                    if num_insert_photo > 0 :
                        logger.info('order_id:{},save mm_ai_order_photo success'.format(order_id)) 
                    else:
                        logger.error('order_id:{},save mm_ai_order_photo fail'.format(order_id)) 
                      
                    # STEP6: 更新mm_order 中的状态
                    num_update_status3 = update_order_status(order_id) 
                    if num_update_status3 == 1 :
                        logger.info('order_id:{},update_status3 success'.format(order_id)) 
                    else:
                        logger.error('order_id:{},update_status3 fail'.format(order_id)) 
                else:
                    logger.error('本次模型预测无结果, order_id:{}'.format(order_id)) 
        else:
            print('本次未抢到单')
        time.sleep(5)  # 休眠5秒 
