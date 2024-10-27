import numpy as np
from tqdm import tqdm
from src.utils.main_utils import *
from src.utils.chat_models import *
from probe_utils import *


def main(model_name="gpt-4o-2024-08-06",
         num_stmts=200,
         probe_setup_id=0,
         mode="stmt",
         data_version="090824_800",
         is_individual_question=False,
         n_devices=2,
         is_together=False):
    if is_individual_question:
        expt_id = f"{mode}-{data_version}-v{probe_setup_id}-num_stmt_{num_stmts}-ind_ques"
    else:
        expt_id = f"{mode}-{data_version}-v{probe_setup_id}-num_stmt_{num_stmts}"
    data_path = f"data/WVS/probe_expts/{data_version}/{expt_id}.jsonl"
    data = load_standard_data(data_path)

    target_model = get_chat_model(model_name,
                                  {"temperature": 0,
                                   "top_p": 1.0,
                                   "num_tokens": 4096,
                                   "n_devices": n_devices,
                                   "is_together": is_together})

    for d in tqdm(data):
        d_probe_prompt = d["probe_prompt"]
        raw_response = target_model.generate(d_probe_prompt)
        while parse_response(raw_response) == "":
            target_model.update_config({"temperature": 0.6})
            raw_response = target_model.generate(d_probe_prompt)
        d["raw_response"] = raw_response
        target_model.update_config({"temperature": 0})

        save_path = data_path.replace(
            "data/WVS/", "results/").replace(f"{expt_id}.jsonl", f"{model_name}/{expt_id}.jsonl")
        write_standard_data(data, save_path)


def main_batch(model_name="gpt-4o-2024-08-06",
               num_stmts=200,
               probe_setup_id=0,
               mode="stmt",
               data_version="probe_data_090524_800",
               n_devices=2):
    expt_id = f"{mode}-{data_version}-v{probe_setup_id}-num_stmt_{num_stmts}"
    data_path = f"data/WVS/probe_expts/{data_version}/{expt_id}.jsonl"
    data = load_standard_data(data_path)

    target_model = get_chat_model(model_name,
                                  {"temperature": 0,
                                   "top_p": 1.0,
                                   "num_tokens": 4096,
                                   "n_devices": n_devices
                                   })

    all_probe_prompts = []
    for d in tqdm(data):
        d_probe_prompt = d["probe_prompt"]
        all_probe_prompts.append(d_probe_prompt)
    raw_responses = target_model.batch_generate(all_probe_prompts)

    for i, raw_response in tqdm(enumerate(raw_responses), total=len(raw_responses)):
        d = data[i]
        while parse_response(raw_response) == "":
            target_model.update_config({"temperature": 0.6})
            raw_response = target_model.generate(d_probe_prompt)
        d["raw_response"] = raw_response
        target_model.update_config({"temperature": 0})

        save_path = data_path.replace(
            "data/WVS/", "results/").replace(f"{expt_id}.jsonl", f"{model_name}/{expt_id}.jsonl")
        write_standard_data(data, save_path)


def main_fix(model_name="gpt-4o-2024-08-06",
             num_stmts=200,
             probe_setup_id=0,
             mode="stmt",
             data_version="090824_800",
             n_devices=2,
             is_together=False):
    expt_id = f"{mode}-{data_version}-v{probe_setup_id}-num_stmt_{num_stmts}"
    prompt_data_path = f"data/WVS/probe_expts/{data_version}/{expt_id}.jsonl"
    prompt_data = load_standard_data(prompt_data_path)

    data_path = prompt_data_path.replace(
        "data/WVS/", "results/").replace(f"{expt_id}.jsonl", f"{model_name}/{expt_id}.jsonl")
    data = load_standard_data(data_path)
    target_model = get_chat_model(model_name,
                                  {"temperature": 0,
                                   "top_p": 1.0,
                                   "num_tokens": 4096,
                                   "n_devices": n_devices,
                                   "is_together": is_together})

    # MAX_RETRIES = 10
    num_failed_d = 0
    all_data_to_save = []
    for i, d in tqdm(enumerate(prompt_data), total=len(prompt_data)):
        if i < len(data):
            d_gen = data[i]
        else:
            d_gen = d
        d_probe_prompt = d_gen["probe_prompt"]

        # num_retries = 0
        if "raw_response" in d_gen:
            raw_response = d_gen["raw_response"]
            # and num_retries < MAX_RETRIES:
            while parse_response(raw_response) == "":
                target_model.update_config({"temperature": 0.6})
                raw_response = target_model.generate(d_probe_prompt)
                # num_retries += 1
            # if num_retries == MAX_RETRIES:
            #     num_failed_d += 1
            d_gen["raw_response"] = raw_response
            target_model.update_config({"temperature": 0})
        else:
            raw_response = target_model.generate(d_probe_prompt)
            d_gen["raw_response"] = raw_response

        all_data_to_save.append(d_gen)
        save_path = data_path.replace(".jsonl", "_fixed.jsonl")
        write_standard_data(all_data_to_save, save_path)

    print(f"Num failed data: {num_failed_d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="gpt-4o-2024-08-06")
    parser.add_argument("--num_stmts", type=int, default=200)
    # parser.add_argument("--num_stmts", type=str,
    # default="social_values,_at
    # titudes_&_stereotypes")
    parser.add_argument("--probe_setup_id", type=int, default=0)
    parser.add_argument("--mode", type=str, default="stmt")
    parser.add_argument("--data_version", type=str,
                        default="090824_800_refined")
    parser.add_argument("--is_individual_question", type=bool, default=False)
    parser.add_argument("--n_devices", type=int, default=2)
    parser.add_argument("--is_together", type=bool, default=False)
    args = parser.parse_args()

    # data_to_gen = ["social_values,_attitudes_&_stereotypes",
    #                "happiness_and_well-being",
    #                "social_capital,_trust_&_organizational_membership",
    #                "economic_values",
    #                "corruption",
    #                "migration",
    #                "security",
    #                "postmaterialist_index",
    #                "science_&_technology",
    #                "religious_values",
    #                "ethical_values_and_norms",
    #                "political_interest_&_political_participation",
    #                "political_culture_&_political_regimes"]

    # for num_stmts in data_to_gen:
    #     main(model_name=args.model_name,
    #         num_stmts=num_stmts,
    #         probe_setup_id=args.probe_setup_id,
    #         mode=args.mode,
    #         data_version=args.data_version,
    #         n_devices=args.n_devices,
    #         is_together=args.is_together)

    main(model_name=args.model_name,
         num_stmts=args.num_stmts,
         probe_setup_id=args.probe_setup_id,
         mode=args.mode,
         data_version=args.data_version,
         n_devices=args.n_devices,
         is_together=args.is_together)


# "claude-3-haiku-20240307"
# "gpt-4-turbo-2024-04-09"
# "google/gemma-2-27b-it",
# "google/gemma-2-9b-it",
# "gpt-4o-2024-08-06",
# "gpt-4o-2024-05-13",
# "gpt-4o-mini-2024-07-18",
# "gpt-4-turbo-2024-04-09",
# "gpt-3.5-turbo-0125",
# "meta-llama/Meta-Llama-3.1-8B-Instruct"
# "meta-llama/Meta-Llama-3.1-70B-Instruct"
