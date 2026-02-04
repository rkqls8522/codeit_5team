
#@title 알약id와 모델에 넣을 id매핑. yolo는 0부터, fast rcnn은 1부터.
# 알약 종류 갯수 먼저 구하고
import json
import glob
import ast
import os

def class_mapping(json_dir, txt_file_name="ClassID.txt"):
    category_map = {}

    json_paths = glob.glob(
        os.path.join(json_dir, "**", "*.json"),
        recursive=True
    )

    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
            for cat in data["categories"]:
                cid = cat["id"]
                if cid not in category_map:
                    category_map[cid] = cat["name"]

    print(f"총 클래스 수: {len(category_map)}")

    # 매핑하기
    class_id_map = {}
    for i, (cid, name) in enumerate(sorted(category_map.items())):
        class_id_map[cid] = {
            "name": name,
            "yolo_id": i,
            "fasterrcnn_id": i + 1
        }

    with open(os.path.join(json_dir, txt_file_name), "w", encoding="utf-8") as f:
        for k, v in class_id_map.items():
            f.write(f"{k} {v}\n")



def read_classID(dir, txt_file_name="ClassID.txt"):
    cat_id = {}

    with open(os.path.join(dir, txt_file_name), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key_str, dict_str = line.split(" ", 1)

            key = int(key_str)
            value_dict = ast.literal_eval(dict_str)

            cat_id[key] = value_dict

    return cat_id