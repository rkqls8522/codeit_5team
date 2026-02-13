# annotation json에서 category id 추출해서 1부터 매핑하는 txt 생성
# Faster R-CNN은 0이 background라서 1부터 시작해야 함

import os
import glob
import json as json_json

def make_classIDtxt(anntation_dir, file_name="class_id"):
    anntation_file_path_list = glob.glob(
        os.path.join(anntation_dir, "**", "*.json"), recursive=True
    )

    json_list = []
    for path in anntation_file_path_list:
        with open(path, "r", encoding="utf-8") as f:
            json_list.append(json_json.load(f))

    annt_list = [
        {
            "categories": json["categories"]
        }
        for json in json_list
    ]

    # category id -> yolo id 매핑 (1부터 시작)
    cat_id = {
        cat_id: idx
        for idx, cat_id in enumerate(
            sorted({annt["categories"][0]["id"] for annt in annt_list}),
            start=1
        )
    }

    # 매핑 txt 저장
    txt_path = os.path.join(anntation_dir, file_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in cat_id.items():
            f.write(f"{k} {v}\n")

    return txt_path, cat_id