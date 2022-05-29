import json
from datetime import datetime, timedelta
import pandas as pd


def get_json(config_path):
    """ [json 파일 받기]
    
    /* */로 주석 처리된 json 파일 받기
    
    Args:
        config_path (str) : json 파일 경로
    
    Returns:
        dict : dictionary 타입의 json 파일
        
    """
    
    with open(config_path, 'r', encoding='utf-8') as f:
        contents = f.read()
        while "/*" in contents:
            preComment, postComment = contents.split('/*', 1)
            contents = preComment + postComment.split('*/', 1)[1]
            
        return json.loads(contents.replace("'", '"'))


def set_config_restrictions(config):
    """ [config의 제약사항 설정]
    
    config의 dataset 양이나 기간 등에서의 제약 사항으로 인한 변경값 설정
    
    Args:
        config (easydict.EasyDict) : config 
    
    Returns:
        easydict.EasyDict : 제약 사항으로 인해 변경된 config
    
    """
    
    if config.dataset.to == None:
        print("* set config.dataset.to to current date")
        print(f"   [config.dataset.to : None -> {datetime.now().strftime('%Y%m%d %H')}]\n")
        config.dataset.to = datetime.now().strftime("%Y%m%d %H")
    
    if config.dataset.count > 2500:
        print("* config.dataset.count must be below 2500")
        print(f"   [config.dataset.count : {config.dataset.count} -> 2000]\n")
        config.dataset.count = 2000

    if datetime.strptime(config.dataset.to, '%Y%m%d %H') > datetime.now():
        print("* config.dataset.to can not be passed current date") 
        print(f"   [config.dataset.to : {config.dataset.to} -> {datetime.now().strftime('%Y%m%d %H')}]\n")
        config.dataset.to = datetime.now().strftime('%Y%m%d %H')

    if datetime.strptime(config.dataset.to, '%Y%m%d %H') < datetime.strptime('20180101', '%Y%m%d'):
        print("* config.dataset.to can not be before 2018.01.01") 
        print(f"   [config.dataset.to : {config.dataset.to} -> 20180101 00]\n")
        config.dataset.to = '20180101 00'

    if config.model.name == 'prophet':
        if config.model.cycle != 1:
            print("* Prophat must train every single cycle")
            print(f"   [config.model.cycle : {config.model.cycle} -> 1]")
            config.model.cycle = 1
    
    return config