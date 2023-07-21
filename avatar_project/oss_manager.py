# -*- coding: utf-8 -*-
# @version:    ai_avator v1.0
# @author:     FanWen
# @license:
# @file:       oss_manager.py
# @time:       2023/06/24
# @Description: 涉及到oss 的操作
# @Others:

import os
import oss2
import time
from logger_manager import init_logger

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")  
logger  = init_logger(log_dir, "main")  

class HandleOSSUtil(object):
    def __init__(self, key_id, key_secret, bucket=None):
        '''
        :param key_id:
        :param key_secret:
        :param bucket: bucket名字，例如：test
        '''
        self.auth = oss2.Auth(key_id, key_secret)
        self.link_url  =  'http://oss-cn-beijing.aliyuncs.com'
        if bucket:
            self.bucket = oss2.Bucket(self.auth, self.link_url, bucket)

    def update_one_file(self, oss_path, file_path):
        '''
        @summary: 将一个文件上传
        @param file_path: 要上传图片的本地地址，如：images/O12023061915104587300002/output/20001_O12023061915104587300001_1.jpg
        @param oss_path:   oss上的output路径: userdata-image-output/20001_O12023061915104587300001_1.jpg
        @update: 增加最大重试次数
        '''
        max_retries = 3 
        retry_count = 0
        while retry_count < max_retries:
            try:
                result = self.bucket.put_object_from_file(oss_path, file_path)
                if result.status == 200:
                    return 1
            except Exception as e:
                retry_count += 1
                logger.error('oss上传发生异常:{},重试第{}次, file_path:{}'.format(str(e),retry_count,file_path))
                time.sleep(1)  
        
        logger.error('oss上传重试三次失败, file_path: {}'.format(file_path))
        return 0

 
    def download_one_file(self, oss_path, save_dir):
        '''
        @summary: 下载单个文件
        @param oss_path: 文件所在的oss地址，例如：test/test.png
        @param save_dir: 要保存在本地的文件目录，例如：/images
        @update: 增加最大重试次数
        '''
        file_name = oss_path.split('/')[-1]
        save_path = os.path.join(save_dir, file_name)
        try_count = 0
        max_retries = 3  
        while try_count < max_retries:
            try:
                result = self.bucket.get_object_to_file(oss_path, save_path)
                if result.status == 200:
                    return 1
            except Exception as e:
                try_count += 1
                logger.error('oss下载发生异常:{}, 重试第{}次, oss_path:{}'.format(str(e),try_count, oss_path))
                time.sleep(1)  

        logger.error('oss下载重试三次失败,oss_path:{}'.format(oss_path))
        return 0
 
    # 下载文件夹中所有文件
    def download_many_file(self, oss_dir, save_dir):
        '''
        @param oss_dir: oss上要下载的文件目录，例如：test/test_result/
        @param save_dir: 要存在本地的文件目录，例如：/images
        @return:
        '''
        obj = oss2.ObjectIterator(self.bucket, prefix=oss_dir)
        # 遍历oss文件夹获取所有的对象列表，i.key是文件的完整路径
        for i in obj:
            # 如果文件是以斜杠结尾的，说明不是文件，则跳过
            if i.key.endswith('/'):
                continue
            # 文件名：文件路径按照斜杠分割取最后一个
            file_name = i.key.split('/')[-1]
            # 下载到的具体路径
            save_path = os.path.join(save_dir, file_name)
            # 从oss下载
            self.bucket.get_object_to_file(i.key, save_path)
