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

    def update_one_file(self, oss_path,file_path):
        '''
        @summary: 将一个文件上传
        @param file_path: 要上传图片的本地地址，如：images/O12023061915104587300002/output/20001_O12023061915104587300001_1.jpg
        @param oss_dir:   oss上的output路径: userdata-image-output/20001_O12023061915104587300001_1.jpg
        @return:
        '''
        result = self.bucket.put_object_from_file(oss_path, file_path)
        if result.status == 200:
            return 1
 
    def update_file(self, oss_dir,file_dir ):
        '''
        @summary: 将一个文件夹下面的所有文件都上传
        @param file_dir: 要上传图片所在的文件夹，例如：/images
        @param oss_dir:  oss上的路径: userdata-image-output
        @return:
        '''
        for i in os.listdir(file_dir):
            if i.endswith(".jpg"):
                # oss上传后的路径
                oss_path = f'{oss_dir}/{i}'
                # 本地文件路径
                file_path = f'{file_dir}/{i}'
                # 进行上传
                self.bucket.put_object_from_file(oss_path, file_path)
            
 
    def download_one_file(self, oss_path, save_dir):
        '''
        @summary: 下载单个文件
        @param oss_path: 文件所在的oss地址，例如：test/test.png
        @param save_dir: 要保存在本地的文件目录，例如：/images
        @return:
        '''
        file_name = oss_path.split('/')[-1]
        save_path = os.path.join(save_dir, file_name)
        result = self.bucket.get_object_to_file(oss_path, save_path)
        if result.status == 200:
            return 1
 
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