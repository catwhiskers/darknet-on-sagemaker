
import sys
import os  
import shutil
import subprocess
import json 
train_bash_command_template = "darknet detector train {} {} {}  -dont_show" 
predict_bash_command_template = "darknet detector test {} {} {} {} -dont_show" 

#sagemaker data 
hyperparameters_file_path = "/opt/ml/input/config/hyperparameters.json"
inputdataconfig_file_path = "/opt/ml/input/config/inputdataconfig.json"
resource_file_path = "/opt/ml/input/config/resourceconfig.json"
data_files_path = "/opt/ml/input/data/"
failure_file_path = "/opt/ml/output/failure"
model_artifacts_path = "/opt/ml/model/"


#path configuration 
path_config_key = "config_path"
path_config_fn = "config.json"
cfg_key = "cfg"
dinfo_key = "dinfo"
weight_key = "weight"

#configuration in configuration 
weight_key = "weight"


def parse_config_path():
    path = os.path.join(data_files_path, path_config_key, path_config_fn)
    jsonf = open(path, 'r')
    import json 
    obj = json.load(jsonf)
    return obj               

import re

def get_configuration_paths():
    obj = parse_config_path() 
    cfg_path = obj[cfg_key]
    dinfo_path = obj[dinfo_key]
    weight_path = obj[weight_key]
    df = open(dinfo_path,'r')
    backup_path = 'backup' 
    for l in df.readlines():
        if weight_key in l:
             backup_search = re.search("backup( *)=(.*)", l)
             backup_path = backup_search.group(2)
    return cfg_path, dinfo_path, weight_path, backup_path


def modify_cfg_based_on_hyperparams(cfg_file_path):
    hyper_params = json.load(open(hyperparameters_file_path, 'r'))
    cfg_contents = []
    cfg_dict = {}
    cfg_file = open (cfg_file_path, 'r')
    for l in cfg_file.readlines():
        if '=' in l:
            toks = l.split('=')
            cfg_dict[toks[0].strip()]=len(cfg_contents)
            cfg_contents.append([toks[0], toks[1]])
        else:
            cfg_contents.append([l])
    print(hyper_params)
    print(type(hyper_params))
    for k,v in hyper_params.items():
        idx = cfg_dict[k]
        cfg_contents[idx] = [k, v+'\n']

    tmp_cfg_path = '/tmp/modified.cfg'
    tmp_cfg = open(tmp_cfg_path, 'w')
    for c in cfg_contents:
        if len(c) == 1:
            tmp_cfg.write(c[0])
        elif len(c) == 2:
            tmp_cfg.write(c[0]+'='+c[1])
    tmp_cfg.close()
    shutil.copy(tmp_cfg_path, cfg_file_path)

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def train():
    cfg_path, dinfo_path, weight_path, backup_path = get_configuration_paths()  
    if os.path.exists(hyperparameters_file_path):
        modify_cfg_based_on_hyperparams(cfg_path)
    train_local(cfg_path,dinfo_path,weight_path)
    if os.path.exists(backup_path):
        copytree(backup_path, model_artifacts_path)

def mk_dir(sub_dir, working_dir=None):
    output_path = None 
    if working_dir: 
        output_path = working_dir+'/'+sub_dir
    else: 
        output_path = sub_dir 
    if not os.path.exists(output_path):
        os.mkdir(output_path) 

def train_local(cfg_path, dinfo_path, weights_path, working_dir=None):
    mk_dir('backup', working_dir)
    train_bash_command = train_bash_command_template.format(dinfo_path, cfg_path, weights_path)
    process = subprocess.Popen(train_bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output, error)
   

def predict_local(cfg_path,dinfo_path, weights_path, test_file, working_dir):
    bash_command = predict_bash_command_template.format(dinfo_path, cfg_path, weights_path, test_file)
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    if os.path.isfile('predictions.jpg'):
        shutil.copyfile('predictions.jpg', working_dir+'/predictions.jpg')
    print(output, error)
   

if __name__ == "__main__":
    if (sys.argv[1] == "train"):
        train()
    elif (sys.argv[1]=="train_local"):
        cfg_path = sys.argv[2]   
        dinfo_path = sys.argv[3]
        weight_path = sys.argv[4]
        working_dir = sys.argv[5]
        train_local(cfg_path, dinfo_path, weight_path, working_dir)
    elif (sys.argv[1]=="predict_local"):
        cfg_path = sys.argv[2]   
        dinfo_path = sys.argv[3]
        weight_path = sys.argv[4]
        test_file = sys.argv[5]
        working_dir = sys.argv[6]
        predict_local(cfg_path, dinfo_path, weight_path, test_file, working_dir)
    else:
        print("Missing required argument 'train, traing_local or predict_local'.", file=sys.stderr)
        sys.exit(1)


