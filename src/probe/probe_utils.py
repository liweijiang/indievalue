import numpy as np
from tqdm import tqdm
from src.utils.main_utils import *
from src.utils.chat_models import *


def parse_response(response):
    try:
        response = response.replace("```", "")
        response = response.replace("json", "")
        response = response.replace("[Your Response]:", "")
        response = response.replace("Response:", "")
        response = response.strip(" ")
        response = response.strip("\n")
        response_dict = json.loads(response)

        for nsg_id in response_dict:
            q_d = response_dict[nsg_id]
            try:
                q_d["choice_grouped_answer_id"] = int(
                    q_d["choice"].replace("NSG", "").split("_s")[-1]) - 1
            except Exception as e:
                return ""
        if len(response_dict) != 39:
            return ""
        return response_dict
    except json.JSONDecodeError:
        # print("=" * 50)
        # print("Error: Unable to parse the response as JSON.")
        # print(response)
        # print("=" * 50)
        return ""
    except KeyError:
        print("Error: The expected keys are not present in the parsed JSON.")
        return ""


def get_accuracy(d, probe_qids):
    d_acc = []
    for qid in probe_qids:
        qid_is_correct = d[f"{qid}_is_correct"]
        d_acc.append(qid_is_correct)
    d_acc = np.mean(d_acc)
    return d_acc
