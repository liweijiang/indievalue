import numpy as np
from src.utils.main_utils import *
from src.utils.chat_models import *


def parse_outputs(raw_outputs):
    clean_option_choices = []
    clean_choice_statements = []
    for output in raw_outputs:
        try:
            clean_option_choice = output.split(": ")[0].lower().strip()
            clean_choice_statement = output.split(": ")[-1].lower().strip()
            if "1" in clean_option_choice:
                clean_option_choice = "option 1"
            elif "2" in clean_option_choice:
                clean_option_choice = "option 2"
            elif "3" in clean_option_choice:
                clean_option_choice = "option 3"
            elif "4" in clean_option_choice:
                clean_option_choice = "option 4"
            elif "5" in clean_option_choice:
                clean_option_choice = "option 5"
            elif "6" in clean_option_choice:
                clean_option_choice = "option 6"
            elif "7" in clean_option_choice:
                clean_option_choice = "option 7"
            elif "8" in clean_option_choice:
                clean_option_choice = "option 8"
            clean_option_choices.append(clean_option_choice)
            clean_choice_statements.append(clean_choice_statement)
        except:
            clean_option_choices.append(None)
            clean_choice_statements.append(None)
    return clean_option_choices, clean_choice_statements


def compute_accuracies(label_option_choices, label_choice_statements, model_option_choices, model_choice_statements, if_ind=True):
    accuracies = []
    for label_option_choice, label_choice_statement, model_option_choice, model_choice_statement in zip(label_option_choices, label_choice_statements, model_option_choices, model_choice_statements):
        if label_option_choice != None and model_option_choice != None and label_option_choice == model_option_choice:
            accuracies.append(1)
        else:
            accuracies.append(0)

    if if_ind:
        return accuracies
    else:
        return np.mean(accuracies)


def main(model_config, eval_data_config):
    split = eval_data_config["split"]
    probe_setup_id = eval_data_config["probe_setup_id"]
    eval_data_version = eval_data_config["data_version"]
    probe_mode = eval_data_config["probe_mode"]
    num_demo = eval_data_config["num_demo"]
    demo_mode = eval_data_config["demo_mode"]
    num_to_eval = eval_data_config["num_to_eval"]

    # if demo_mode == "demo":
    #     eval_data_path = f"data/WVS/training_expts/{split}/v2/{demo_mode}_{probe_mode[0]}1_{eval_data_version}_v{probe_setup_id}.jsonl"
    # else:
    #     eval_data_path = f"data/WVS/training_expts/{split}/v1/{demo_mode[0]}{num_demo}_{probe_mode[0]}1_{eval_data_version}_v{probe_setup_id}.jsonl"

    if demo_mode == "demo":
        eval_data_path = f"/data/{demo_mode}_{probe_mode[0]}1_{eval_data_version}_v{probe_setup_id}.jsonl"
    else:
        eval_data_path = f"/data/{demo_mode[0]}{num_demo}_{probe_mode[0]}1_{eval_data_version}_v{probe_setup_id}.jsonl"

    # eval_data_path = f"/data/{demo_mode[0]}{num_demo}_{probe_mode[0]}1_{eval_data_version}_v{probe_setup_id}.jsonl"
    eval_data = load_standard_data(eval_data_path)

    target_model = get_chat_model(model_config["model_name"],
                                  {"temperature": model_config["temperature"],
                                   "top_p": model_config["top_p"],
                                   "num_tokens": model_config["num_tokens"],
                                   "n_devices": model_config["n_devices"],
                                   "is_show_prompt": model_config["is_show_prompt"]})

    if num_to_eval != None:
        if "meta-llama" in model_config["model_name"]:
            eval_prompts = [
                d["prompt"] + "\nPlease only response with the Option ID without any explanation." for d in eval_data[:num_to_eval]]
        else:
            eval_prompts = [d["prompt"] for d in eval_data[:num_to_eval]]
        eval_responses = [d["response"] for d in eval_data[:num_to_eval]]
        eval_data = eval_data[:num_to_eval]
    else:
        eval_prompts = [d["prompt"] for d in eval_data]
        eval_responses = [d["response"] for d in eval_data]
        eval_data = eval_data
    label_option_choices, label_choice_statements = parse_outputs(
        eval_responses)

    eval_outputs = target_model.batch_generate(eval_prompts)

    model_option_choices, model_choice_statements = parse_outputs(eval_outputs)

    accuracies = compute_accuracies(
        label_option_choices, label_choice_statements, model_option_choices, model_choice_statements, if_ind=False)

    for l, ls, m, ms in zip(label_option_choices, label_choice_statements, model_option_choices, model_choice_statements):
        print("label:", l, ls)
        print("model:", m, ms)
        print("-"*100)

    print(f"Accuracy: {np.mean(accuracies)}")

    raw_output_save_path = "/results/raw_outputs.jsonl"
    for i, d in enumerate(eval_data):
        d["model_choice_id"] = model_option_choices[i]
        d["model_choice_statement"] = model_choice_statements[i]
        d["label_choice_id"] = label_option_choices[i]
        d["label_choice_statement"] = label_choice_statements[i]
    write_standard_data(eval_data, raw_output_save_path)

    config_save_path = "/results/metrics.json"
    configs = {
        "model_config": model_config,
        "eval_data_config": eval_data_config,
        "accuracy": np.mean(accuracies),
        "std": np.std(accuracies),
    }
    save_json(configs, config_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="../indie_models/150-150")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--num_tokens", type=int, default=4096)
    parser.add_argument("--n_devices", type=int, default=2)
    parser.add_argument("--is_show_prompt", action="store_true")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--probe_setup_id", type=int, default=0)
    parser.add_argument("--data_version", type=str, default="090824_800")
    parser.add_argument("--num_demo", type=int, default=200)
    parser.add_argument("--demo_mode", type=str, default="polar")
    parser.add_argument("--probe_mode", type=str, default="polar")
    parser.add_argument("--num_to_eval", type=int, default=None)
    args = parser.parse_args()

    model_config = {
        "model_name": args.model_name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_tokens": args.num_tokens,
        "n_devices": args.n_devices,
        "is_show_prompt": args.is_show_prompt
    }
    eval_data_config = {
        "split": args.split,
        "probe_setup_id": args.probe_setup_id,
        "data_version": args.data_version,
        "num_demo": args.num_demo,
        "demo_mode": args.demo_mode,
        "probe_mode": args.probe_mode,
        "num_to_eval": args.num_to_eval,
    }
    main(model_config, eval_data_config)
