# coding: utf-8

###########################################################
# imagenet_2012_challenge_label_map_proto.pbtxt：
# 是1000种分类结果编号(target_class)，例如：target_class: 450；
# 以及 分类结果编号对应的语言描述的编号(target_class_string)，例如：target_class_string: "n01443537"
id_string_path_file = "..\\InceptionV3_Data\\imagenet_2012_challenge_label_map_proto.pbtxt"

#1. 读取imagenet_2012_challenge_label_map_proto.pbtxt文件
id_string_fileHandle = open(id_string_path_file, mode='r')
# 最终要存储到字典id_to_string中：{target_class_id: target_class_string}
id_to_string = {}
#逐行处理
for line in id_string_fileHandle.readlines():
    #去掉换行符
    line = line.strip('\n')
    if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
    if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        #target_class_string格式是"n01443537"，需要去除首尾引号
        id_to_string[target_class] = target_class_string[1:-1]

id_string_fileHandle.close()

###########################################################
# imagenet_synset_to_human_label_map.txt：
# 是 分类结果编号对应的语言描述的编号(target_class_string) 和 分类结果编号对应的语言描述
# 例如：n01443537  goldfish, Carassius auratus
string_desc_path_file = "..\\InceptionV3_data\\imagenet_synset_to_human_label_map.txt"

#2. 读取imagenet_synset_to_human_label_map.txt文件
string_desc_fileHandle = open(string_desc_path_file, mode='r')
# 最终要存储到字典string_to_desc中：{target_class_string: target_class_desc}
string_to_desc = {}
#逐行处理
for line in string_desc_fileHandle.readlines():
    #去掉换行符
    line = line.strip('\n')
    #每行两个字段间按tab分隔
    list = line.split('\t')
    #keyvalue[0]是target_class_string，keyvalue[1]是target_class_desc，存入字典string_to_desc
    string_to_desc[list[0]] = list[1]

string_desc_fileHandle.close()

###########################################################
# 3.生成想要的1000种分类结果编号(target_class) id 和 分类结果编号对应的语言描述desc的字典
id_to_desc = {}
for key,value in id_to_string.items():
    id_to_desc[key] = string_to_desc[value]

# 4.将步骤3生成的字典存储为文件，后续直接使用
id_desc_path_file = "id_desc.txt"
id_desc_fileHandle = open(id_desc_path_file, mode='w')
id_desc_fileHandle.write(str(id_to_desc))
id_desc_fileHandle.close()