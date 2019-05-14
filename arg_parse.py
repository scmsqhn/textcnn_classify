#!/usr/bin/env python3
import argparse

"""
python 文件处理传入参数
"""

parser = argparse.ArgumentParser(description="arg_parser")
param_lst_full_name = []
param_lst_one_char = []
param_lst_full_name.extend(["all","del"])
param_lst_one_char.extend(["a","d"])

"""
每个全名对应一个单字命令a all<==>a
"""

def init_parm_lst():
    for para in param_lst_full_name:
        print("--%s"%para)
        parser.add_argument("--%s"%para,type=str,default=para)
    for para in param_lst_one_char:
        parser.add_argument("-%s"%para,type=str,default=para)
    return parser

def show_para_lst(parser):
    #args = parser.parse_args
    print("\n-- 包括以下参数:")
    for i in param_lst_full_name:
        print("--%s"%i)
    print("\n- 包括以下参数:")
    for i in param_lst_one_char:
        print("-%s"%i)

parser = init_parm_lst()
show_para_lst(parser)
