# -*- coding: utf-8 -*-
# @version:    ai_avator v1.0
# @author:     FanWen
# @license:
# @file:       mysql_manager.py
# @time:       2023/06/23
# @Description: 涉及到mysql 的操作
# @update: 2023/07/02 修改连接池的使用

import pymysql as MySQLdb
from pymysql.cursors import DictCursor
from dbutils.pooled_db import PooledDB

import traceback
import json
import sys
sys.path.append("..") 

config = json.load(open("conf.json", encoding='utf-8'))

class MysqlManager(object):
    '''
    :Description: 涉及到的mysql操作  
    '''
    __pool = None

    def __init__(self):
        """
        数据库创建连接池
        """
        self.mysql_dict = config["mysql_databases"]
        if MysqlManager.__pool is None:
            MysqlManager.__pool = PooledDB(
                creator=MySQLdb,
                cursorclass=DictCursor,
                mincached=1,
                maxcached=5,
                # maxusage=10,
                host=self.mysql_dict["DBHOST"],
                # port=self.mysql_dict["DBPORT"],
                user=self.mysql_dict["DBUSER"],
                passwd=self.mysql_dict["DBPWD"],
                db=self.mysql_dict["database"])         

    def __query(self, sql, param=None):
        with self.__pool.connection() as conn:
            cursor = conn.cursor()
            if param is None:
                count = cursor.execute(sql)
            else:
                count = cursor.execute(sql, param)
            cursor.close()
            conn.commit()  # 提交事务
            return count

    def update(self, sql, param=None):
        """
        @summary: 更新数据表记录
        @param sql: ＳＱＬ格式及条件，使用(%s,%s)
        @param param: 要更新的  值 tuple/list
        @return: count 受影响的行数
        """
        return self.__query(sql, param)
 
    def getAll(self, sql, param=None):
        """
        @summary: 执行查询，并取出所有结果集
        @param sql:查询ＳＱＬ，如果有查询条件，请只指定条件列表，并将条件值使用参数[param]传递进来
        @param param: 可选参数，条件列表值（元组/列表）
        @return: result list/boolean 查询到的结果集
        """
        with self.__pool.connection() as conn:
            cursor = conn.cursor()
            if param is None:
                count = cursor.execute(sql)
            else:
                count = cursor.execute(sql, param)
            if count > 0:
                result = cursor.fetchall()
            else:
                result = False

            cursor.close() 
            return result


    def getOne(self, sql, param=None):
        """
        @summary: 执行查询，并取出第一条
        @param sql:查询ＳＱＬ，如果有查询条件，请只指定条件列表，并将条件值使用参数[param]传递进来
        @param param: 可选参数，条件列表值（元组/列表）
        @return: result list/boolean 查询到的结果集
        """
        with self.__pool.connection() as conn:
            cursor = conn.cursor() 
            if param is None:
                count = cursor.execute(sql)
            else:
                count = cursor.execute(sql, param)
            if count > 0:
                result = cursor.fetchone()
            else:
                result = False
            cursor.close()
            return result

    def insertOne(self, sql):
        """
        @summary: 向数据表插入一条记录
        @param sql:要插入的ＳＱＬ格式
        @param value:要插入的记录数据tuple/list
        @return: insertId 受影响的行数
        """
        with self.__pool.connection() as conn: 
            cursor = conn.cursor() 
            res = cursor.execute(sql)
            cursor.close()
            return res

    def insertMany(self, sql, values):
        """
        @summary: 向数据表插入多条记录
        @param sql:要插入的ＳＱＬ格式
        @param values:要插入的记录数据tuple(tuple)/list[list]
        @return: count 受影响的行数
        """
        with self.__pool.connection() as conn: 
            cursor = conn.cursor() 
            count  = cursor.executemany(sql, values)
            cursor.close()
            conn.commit()  # 提交事务
            return count

    def update(self, sql, param=None):
        """
        @summary: 更新数据表记录
        @param sql: ＳＱＬ格式及条件，使用(%s,%s)
        @param param: 要更新的  值 tuple/list
        @return: count 受影响的行数
        """
        return self.__query(sql, param)

    def delete(self, sql, param=None):
        """
        @summary: 删除数据表记录
        @param sql: ＳＱＬ格式及条件，使用(%s,%s)
        @param param: 要删除的条件 值 tuple/list
        @return: count 受影响的行数
        """
        return self.__query(sql, param)


