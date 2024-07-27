import datetime
import os
import zipfile
from io import BytesIO

import requests

chattts_service_host = os.environ.get("CHATTTS_SERVICE_HOST", "localhost")
chattts_service_port = os.environ.get("CHATTTS_SERVICE_PORT", "8014")

CHATTTS_URL = f"http://{chattts_service_host}:{chattts_service_port}/generate_voice"

WORDS = """
设备综合效率（OEE）是一种衡量生产设备实际生产能力相对于理论产能比率的独立测量工具。在生产管理中，设备综合效率（Overall Equipment Effectiveness，简称OEE）扮演着重要角色。每一个生产设备都有其理论最大产出，但实际生产中往往因多种干扰和质量损耗而无法达到这一理论值。OEE作为衡量设备实际生产效率与理论生产效率之间差距的重要指标，通过准确跟踪实现“完美生产”的进度，支持全员生产维护（TPM）计划，帮助识别生产过程中的损失和浪费，为改进提供方向。

OEE由可用率、表现性指数和质量指数三个关键要素组成。可用率评估由于设备故障、原材料短缺或生产方法改变等原因导致的停工损失； 表现性指数则评价生产速度上的损失，包括设备磨损、材料不合格或操作失误等因素导致生产未能以最大速度进行； 而质量指数则关注质量损失，反映未达到质量要求的产品情况。这三个维度共同构成了OEE的总体框架，使其成为全面客观反映生产效率现状的强有力工具。

OEE的价值在于它能够系统地揭示生产过程中的改进空间。通过分析影响OEE的各个因素（如停机、速度损失和缺陷等），可以找出生产的瓶颈、低效环节和具体的改进措施。这种基于数据的方法使企业能够在流程优化和资源分配方面做出更加明智的决策。此外，长期的使用OEE工具，企业可以轻松地找到影响生产效率的瓶颈，并进行改进和跟踪，从而提升生产效率，避免不必要的耗费。

OEE不仅能够反映出设备的效率，还能帮助管理者发现并减少一般制造业常见的六大损失，即停机损失、换装调试损失、暂停机损失、减速损失、启动过程次品损失和生产正常运行时产生的次品损失。这些损失的存在严重阻碍了设备效率的发挥，因此利用OEE对这些损失进行识别和分析，采取针对性的改善措施，对于提升整体生产效率至关重要。

综上所述，OEE作为一种衡量设备生产效率的关键指标，通过系统的分析和改进，能够帮助企业显著提高生产效率和产品质量，降低生产成本。在当前激烈竞争的市场环境下，有效应用OEE将成为制造企业持续改进、保持竞争优势的重要手段。
"""
# main infer params
body = {
    "text": [
        "四川美食确实以辣闻名，但也有不辣的选择。",
        "比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。",
        WORDS
    ],
    "stream": False,
    "lang": None,
    "skip_refine_text": True,
    "refine_text_only": False,
    "use_decoder": True,
    "audio_seed": 12345678,
    "text_seed": 87654321,
    "do_text_normalization": True,
    "do_homophone_replacement": False,
}

# refine text params
params_refine_text = {
    "prompt": "",
    "top_P": 0.7,
    "top_K": 20,
    "temperature": 0.7,
    "repetition_penalty": 1,
    "max_new_token": 384,
    "min_new_token": 0,
    "show_tqdm": True,
    "ensure_non_empty": True,
    "stream_batch": 24,
}
body["params_refine_text"] = params_refine_text

# infer code params
params_infer_code = {
    "prompt": "[speed_5]",
    "top_P": 0.1,
    "top_K": 20,
    "temperature": 0.3,
    "repetition_penalty": 1.05,
    "max_new_token": 2048,
    "min_new_token": 0,
    "show_tqdm": True,
    "ensure_non_empty": True,
    "stream_batch": True,
    "spk_emb": None,
}
body["params_infer_code"] = params_infer_code


try:
    response = requests.post(CHATTTS_URL, json=body)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content), "r") as zip_ref:
        # save files for each request in a different folder
        dt = datetime.datetime.now()
        ts = int(dt.timestamp())
        tgt = f"./output/{ts}/"
        os.makedirs(tgt, 0o755)
        zip_ref.extractall(tgt)
        print("Extracted files into", tgt)

except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
