# -*- coding: utf-8 -*-
# @version:    ai_avator v1.0
# @author:     FanWen
# @license:
# @file:       logger_manager.py
# @time:       2023/6/24
# @Description: 涉及到日志的操作
# @warnings:   logging 是线程安全的,但多个进程往同一个文件写日志不是安全的

from logging import Logger
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler
import logging
import os
import time

class infoFilter(logging.Filter):
    def filter(self, record):
        if record.levelno > logging.INFO:
            return False
        return True

class warnFilter(logging.Filter):
    def filter(self, record):
        if record.levelno == logging.WARNING:
            return True
        return False

class CommonTimedRotatingFileHandler(TimedRotatingFileHandler):
    '''修改 TimedRotatingFileHandler,使文件分隔不受多进程影响'''

    @property
    def dfn(self):
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        dfn = self.rotation_filename(self.baseFilename + "." +
                                     time.strftime(self.suffix, timeTuple))
        return dfn

    def shouldRollover(self, record):
        """
        是否应该执行日志滚动操作：
        1、存档文件已存在时，执行滚动操作
        2、当前时间 >= 滚动时间点时，执行滚动操作
        """
        dfn = self.dfn
        t = int(time.time())
        if t >= self.rolloverAt or os.path.exists(dfn):
            return 1
        return 0

    def doRollover(self):
        """
        执行滚动操作
        1、文件句柄更新
        2、存在文件处理
        3、备份数处理
        4、下次滚动时间点更新
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        # get the time that this sequence started at and make it a TimeTuple

        dfn = self.dfn

        # 存档log 已存在处理
        if not os.path.exists(dfn):
            self.rotate(self.baseFilename, dfn)

        # 备份数控制
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)

        # 延迟处理
        if not self.delay:
            self.stream = self._open()

        # 更新滚动时间点
        currentTime = int(time.time())
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval

        # If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'midnight'
                or self.when.startswith('M')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            dstNow = time.localtime(currentTime)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:  # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                newRolloverAt += addend
        self.rolloverAt = newRolloverAt


def init_logger(logdir, logger_name):
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if logger_name not in Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)  #ff:log 等级总开关
        datefmt = "%Y-%m-%d %H:%M:%S"
        format_str = "[%(asctime)s]: PID:%(process)d thread:%(thread)d file:%(filename)s line:%(lineno)s func:%(funcName)s " \
        "%(levelname)s %(message)s"
        # %(lineno)s ff:code line
        formatter = logging.Formatter(format_str, datefmt)

        # use logging
        # screen handler of level info 输出到控制台
        handler = StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        ##ff

        ch = CommonTimedRotatingFileHandler(
            os.path.join(logdir, "warning.log"),
            when='midnight',
            backupCount=50)
        ch.setFormatter(formatter)
        #ch.setLevel(logging.WARNING)
        ch.addFilter(warnFilter())
        logger.addHandler(ch)

        # file handler of level info ## ff 根据时间自动切分
        info_handler = CommonTimedRotatingFileHandler(
            os.path.join(logdir, "info.log"), when='midnight', backupCount=50)
        info_handler.setFormatter(formatter)
        #info_handler.setLevel(logging.INFO)
        info_handler.addFilter(infoFilter())
        logger.addHandler(info_handler)

        # file handler of level error
        err_handler = CommonTimedRotatingFileHandler(
            os.path.join(logdir, 'error.log'), when='midnight', backupCount=50)
        err_handler.setFormatter(formatter)
        err_handler.setLevel(logging.ERROR)
        logger.addHandler(err_handler)

    logger = logging.getLogger(logger_name)
    return logger


log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs") 
log_name = "main"
logger = init_logger(log_dir, log_name)
