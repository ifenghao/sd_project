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
handle_oss_util = HandleOSSUtil(key_id=config["oss_config"]["key_id"], 
                                key_secret=config["oss_config"]["key_secret"], 
                                bucket=config["oss_config"]["bucket_name"]) 

def grab_order(): 
    ''':Description:抢单'''
    mysql_manager_conn = MysqlManager() 
    select_sql = "SELECT * FROM mm_order WHERE order_status = 1 AND is_deleted = 0 ORDER BY create_time ASC LIMIT 1"
    result     = mysql_manager_conn.getOne(select_sql)

    if result:
        order_id   = result['order_id']
        update_sql = "UPDATE mm_order SET order_status = 2 WHERE order_id = %s AND order_status = 1 "
        affected_rows = mysql_manager_conn.update(update_sql, (order_id,))
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
    insert_sql = "INSERT INTO mm_ai_order_photo (user_id, order_id, style_code, photo_url) VALUES (%s, %s, %s, %s)"
    values = []
    for style_code, photo_urls in output_dict.items():
        for photo_url in photo_urls:
            values.append((user_id, order_id, style_code, photo_url))      
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
                local_input_path = processor.prepare_paths()

                num_photo = 0 
                for photo_info in photos_info : 
                    oss_input_file    = photo_info['photo_url']  
                    downloadoss_result= handle_oss_util.download_one_file(oss_input_file, local_input_path)
                    if downloadoss_result ==1:
                        num_photo+=1
                logger.info('order_id:{},共下载{}张图片'.format(order_id,str(num_photo)))
                    
                #TODO1: STEP3 train model and predict
                local_output_dict = processor.process()
                print(local_output_dict)
                logger.info('order_id:{},模型训练和预测结束'.format(order_id)) 
                
                # STEP4: 上传oss 并获取url  
                oss_output_dict   = defaultdict(list)
                oss_dir = config["oss_config"]["output_file"] 
                for style,local_output in local_output_dict.items():
                    for file_path in local_output:
                        file_name = file_path.split('/')[-1]
                        oss_path  = f'{oss_dir}/{file_name}'
                        uploadoss_result = handle_oss_util.update_one_file(oss_path,file_path)
                        if uploadoss_result :
                            oss_output_dict[style].append(oss_path) 
                
                # STEP5: 将结果url 存入 mm_ai_order_photo
                num_insert_photo = insert_ai_order_photo(user_id, order_id, oss_output_dict)
                if num_insert_photo > 0 :
                    logger.info('order_id:{}, save mm_ai_order_photo success'.format(order_id)) 
                else:
                    logger.error('order_id:{},save mm_ai_order_photo fail'.format(order_id)) 
                # STEP6: 更新mm_order 中的状态
                num_update_status3 = update_order_status(order_id) 
                if num_update_status3 == 1 :
                    logger.info('order_id:{},update_status3 success'.format(order_id)) 
                else:
                    logger.error('order_id:{},update_status3 fail'.format(order_id)) 
        else:
            logger.info('本次未抢到单')
        time.sleep(5)  # 休眠5秒 