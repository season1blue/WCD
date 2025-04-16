task_to_image_path = {
    "aokvqa": "./data/aokvqa/images",
    "gqa": "/ai/teacher/ssz/all_data/mqa/GQA/images",
    "mvsa_m": "/ai/teacher/ssz/all_data/msa/MVSA_Multiple/data",
    "mvsa_s": "/ai/teacher/ssz/all_data/msa/MVSA_Single/data",
    "textvqa": "/ai/teacher/ssz/all_data/mqa/TextVQA/train_val",
    "docvqa": "./data/docvqa/images",
    "pope": "./data/pope/images",
    "vstar": "./data/vstar/images",
    "vqav2": "./data/vqav2/images",
}

task_to_question_path = {
    "aokvqa": "./data/aokvqa/data.json",
    "gqa": "/ai/teacher/ssz/all_data/mqa/GQA/jsons/balanced_evalset_list.json",
    "mvsa_m": "/ai/teacher/ssz/all_data/msa/MVSA_Multiple/test.json",
    "mvsa_s": "/ai/teacher/ssz/all_data/msa/MVSA_Single/test.json",
    "textvqa": "/ai/teacher/dkc/Assets/TextVQA/TextVQA_0.5.1_val.json",
    "docvqa": "./data/docvqa/data.json",
    "pope": "./data/pope/data.json",
    "vstar": "./data/vstar/data.json",
    "vqav2": "./data/vqav2/data.json",
}

model_to_fullname = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "blip": "Salesforce/instructblip-vicuna-7b",
    "qwen2_5": "Qwen/Qwen2.5-VL-3B-Instruct",
}