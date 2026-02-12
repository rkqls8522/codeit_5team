# 데이터 내 많은 클래스 수를 가진 학습 데이터 파일 이름을 알아 내기 위한 함수
import glob
import os
import json as json_json

def get_file_name_YOLO(anntation_dir, image_dir, sls_num = 0, els_num=None, max_len = 20, load_exts = ("*.jpg", "*.png", "*.jpeg")):

    anntation_file_path_list = glob.glob(os.path.join(anntation_dir,  "**"), recursive=True)
    annt_p_list = [p for p in anntation_file_path_list if os.path.splitext(p)[1]==".json"]

    
    json_list = []
    for i in range(len(annt_p_list)):
        with open(annt_p_list[i], "r", encoding="utf-8") as f:
            json_list.append(json_json.load(f))


    img_p_list = []
    for ext in load_exts:
        img_p_list.extend(
            glob.glob(os.path.join(image_dir, "**", ext), recursive=True)
        )

    id_dict = {}
    for idd in json_list:
        if idd["categories"][0]["id"] not in id_dict:
            id_dict[idd["categories"][0]["id"]] = 1
        else:
            id_dict[idd["categories"][0]["id"]] += 1

    id_dict2 = {}
    for idd in json_list:
        if f"{idd["categories"][0]["id"]}: {idd["categories"][0]["name"]}" not in id_dict2:
            id_dict2[f"{idd["categories"][0]["id"]}: {idd["categories"][0]["name"]}"] = 1
        else:
            id_dict2[f"{idd["categories"][0]["id"]}: {idd["categories"][0]["name"]}"] += 1
    
    if els_num == None:
        sls_list = sorted(id_dict.items(), key=lambda x: x[1])[sls_num: ]
        sls_list2 = sorted(id_dict2.items(), key=lambda x: x[1])[sls_num: ]
    else:
        sls_list = sorted(id_dict.items(), key=lambda x: x[1])[sls_num:els_num]
        sls_list2 = sorted(id_dict2.items(), key=lambda x: x[1])[sls_num:els_num]

    print(f"data 사용 클래스 목록:")
    for namen in sls_list2:
        print(namen)
    print()
    for namen in sls_list:
        print(namen)
    print()
    sls_list_class = [i[0] for i in sls_list]



    file_list = []

    for path in img_p_list:
        flag = False
        flag2 = True
        for annt in json_list:
            if (annt["categories"][0]["id"] in sls_list_class) and (os.path.basename(path) == annt["images"][0]["file_name"]):
                flag = True
            if (annt["categories"][0]["id"] not in sls_list_class) and flag and (os.path.basename(path) == annt["images"][0]["file_name"]):
                flag2 = False

        if flag and flag2:
            file_list.append(os.path.basename(path))
            if len(file_list) >= max_len:
                return file_list

    return file_list
