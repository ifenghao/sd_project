# -*- coding: utf-8 -*-
# @version:    ai_avator v1.0
# @author:     FanWen
# @license:
# @file:       mysql_manager.py
# @time:       2023/06/23
# @Description: 涉及到mysql 的操作
# @Others:

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
        数据库构造函数，从连接池中取出连接，并生成操作游标
        """
        self.mysql_dict = config["mysql_databases"]
        self._conn = MysqlManager.__getConn(self.mysql_dict)
        self._cursor = self._conn.cursor()

    @staticmethod
    def __getConn(mysql_dict):
        """
        @summary: 静态方法，从连接池中取出连接
        @return MySQLdb.connection
        """
        if MysqlManager.__pool is None:
            MysqlManager.__pool = PooledDB(
                creator=MySQLdb,
                cursorclass=DictCursor,
                mincached=1,
                maxcached=50,
                maxusage=10,
                host=mysql_dict["DBHOST"],
                # port=mysql_dict["DBPORT"],
                user=mysql_dict["DBUSER"],
                passwd=mysql_dict["DBPWD"],
                db=mysql_dict["database"])

        return MysqlManager.__pool.connection()

    def autocommit(self, value):
        """
        @summary: 设置是否自动提交事务
        @param value: bool类型，True表示自动提交，False表示手动提交
        """
        self._conn.autocommit(value)

    def __query(self, sql, param=None):
        if param is None:
            count = self._cursor.execute(sql)
        else:
            count = self._cursor.execute(sql, param)
        return count

    def excfun(self, sql):
        try:
            self._cursor.execute(sql)
        except:
            traceback.print_exc()
            self._conn.rollback()
        finally:
            logger.error('mysql 出现问题')
            self._cursor.close()
            self._conn.close()

    def countText(self, sql, param=None):
        if param is None:
            count = self._cursor.execute(sql)
        else:
            count = self._cursor.execute(sql, param)
        if count > 0:
            result = self._cursor.fetchone()
        else:
            result = 0
        return result

    def getAll(self, sql, param=None):
        """
        @summary: 执行查询，并取出所有结果集
        @param sql:查询ＳＱＬ，如果有查询条件，请只指定条件列表，并将条件值使用参数[param]传递进来
        @param param: 可选参数，条件列表值（元组/列表）
        @return: result list/boolean 查询到的结果集
        """
        if param is None:
            count = self._cursor.execute(sql)
        else:
            count = self._cursor.execute(sql, param)
        if count > 0:
            result = self._cursor.fetchall()
        else:
            result = False
        return result

    def getOne(self, sql, param=None):
        """
        @summary: 执行查询，并取出第一条
        @param sql:查询ＳＱＬ，如果有查询条件，请只指定条件列表，并将条件值使用参数[param]传递进来
        @param param: 可选参数，条件列表值（元组/列表）
        @return: result list/boolean 查询到的结果集
        """
        if param is None:
            count = self._cursor.execute(sql)
        else:
            count = self._cursor.execute(sql, param)
        if count > 0:
            result = self._cursor.fetchone()
        else:
            result = False
        return result

    def getMany(self, sql, num, param=None):
        """
        @summary: 执行查询，并取出num条结果
        @param sql:查询ＳＱＬ，如果有查询条件，请只指定条件列表，并将条件值使用参数[param]传递进来
        @param num:取得的结果条数
        @param param: 可选参数，条件列表值（元组/列表）
        @return: result list/boolean 查询到的结果集
        """
        if param is None:
            count = self._cursor.execute(sql)
        else:
            count = self._cursor.execute(sql, param)
        if count > 0:
            result = self._cursor.fetchmany(num)
        else:
            result = False
        return result

    # @ErrorHandler.error_handler
    def insertOne(self, sql):
        """
        @summary: 向数据表插入一条记录
        @param sql:要插入的ＳＱＬ格式
        @param value:要插入的记录数据tuple/list
        @return: insertId 受影响的行数
        """
        # return self.excfun(sql)
        return self._cursor.execute(sql)

    def insertMany(self, sql, values):
        """
        @summary: 向数据表插入多条记录
        @param sql:要插入的ＳＱＬ格式
        @param values:要插入的记录数据tuple(tuple)/list[list]
        @return: count 受影响的行数
        """
        count = self._cursor.executemany(sql, values)
        return count

    def __getInsertId(self):
        """
        获取当前连接最后一次插入操作生成的id,如果没有则为０
        """
        self._cursor.execute("SELECT @@IDENTITY AS id")
        result = self._cursor.fetchall()
        return result[0]['id']

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

    def begin(self):
        """
        @summary: 开启事务
        """
        self._conn.autocommit(0)
        # self._conn.autocommit(1)

    def end(self, option='commit'):
        """
        @summary: 结束事务
        """
        if option == 'commit':
            self._conn.commit()
        else:
            self._conn.rollback()

    def dispose(self, isEnd=1):
        """
        @summary: 释放连接池资源
        """
        if isEnd == 1:
            self.end('commit')
        else:
            self.end('rollback')
        self._cursor.close()
        self._conn.close()


if __name__ == "__main__":
    config = json.load(open("../conf.json", encoding='utf-8'))

    conn = MysqlManager("")
    SQL = '''SELECT * FROM mm_order WHERE order_status = 1 AND is_deleted = 0;)
          '''
    conn.getOne(SQL)
    result = mysql_manager.getOne(sql)
    conn.dispose()