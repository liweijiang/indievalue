from src.utils.main_utils import *
from data.WVS.WVS_conversion import *
from transformers import AutoTokenizer


def load_train_template(train_template_version=1):
    prompt_template_path = f"data/util_prompts/train_template_v{train_template_version}.txt"
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read()
    return prompt_template


def format_train_data(demo_statements, probe_statement_options, probe_gaid, template_version=1):
    prompt_template = load_train_template(template_version)

    demo_statements_string = ""
    for demo_statement in demo_statements:
        demo_statements_string += f"# {demo_statement[0].upper()}{demo_statement[1:]}\n"

    probe_statement_string = ""
    for i, probe_statement in enumerate(probe_statement_options):
        probe_statement_string += f"Option {i+1}: {probe_statement}\n"

    response = f"Option {probe_gaid + 1}: {probe_statement_options[probe_gaid]}"

    prompt = prompt_template.replace(
        "{known_statements}", demo_statements_string)
    prompt = prompt.replace("{probe_statements}", probe_statement_string)
    return prompt, response


def get_probe_statement_options(probe_qid, probe_metadata_map):
    return probe_metadata_map[probe_qid]["converted_statements"]


def get_human_id_to_human_data_map(human_label_data):
    human_id_to_human_data_map = {}
    for human_label_d in human_label_data:
        human_id = human_label_d["D_INTERVIEW"]
        human_id_to_human_data_map[human_id] = human_label_d
    return human_id_to_human_data_map


def load_probe_data(data_version, is_refined):
    if is_refined:
        data_path = f"data/WVS/human_labels/probe_data/{data_version}_refined.jsonl"
    else:
        data_path = f"data/WVS/human_labels/probe_data/{data_version}.jsonl"
    probe_data = load_standard_data(data_path)
    return {data_d["D_INTERVIEW"]: data_d for data_d in probe_data}


def load_data(demo_mode, probe_mode, probe_setup_id, data_version):
    refined_probe_human_label_data = load_probe_data(
        data_version, is_refined=True)
    polar_probe_human_label_data = load_probe_data(
        data_version, is_refined=False)
    probe_human_ids = list(refined_probe_human_label_data.keys())

    random.seed(42)
    random.shuffle(probe_human_ids)

    split_probe_human_ids = {"test": probe_human_ids[:int(len(probe_human_ids) * 0.5)],
                             "val": probe_human_ids[int(len(probe_human_ids) * 0.5):]}

    refined_statements_metadata_map = get_statements_metadata_map(
        is_refined=True)
    polar_statements_metadata_map = get_statements_metadata_map(
        is_refined=False)
    all_probe_QIDs = get_all_probe_qids(probe_setup_id=probe_setup_id)

    if demo_mode == "polar":
        demo_human_label_data = polar_probe_human_label_data
        demo_metadata_map = polar_statements_metadata_map
    else:
        demo_human_label_data = refined_probe_human_label_data
        demo_metadata_map = refined_statements_metadata_map
    if probe_mode == "polar":
        probe_human_label_data = polar_probe_human_label_data
        probe_metadata_map = polar_statements_metadata_map
    else:
        probe_human_label_data = refined_probe_human_label_data
        probe_metadata_map = refined_statements_metadata_map

    all_statements_QIDs = get_all_statements_QIDs()

    return demo_human_label_data, demo_metadata_map, probe_human_label_data, probe_metadata_map, all_probe_QIDs, split_probe_human_ids, all_statements_QIDs


def compile_evaluation_data_units(
        data_version="090824_800",
        template_version=2,
        probe_setup_id=0,
        demo_mode="polar",
        num_demo=200,
        probe_mode="polar",
        eval_data_version="v1",
):
    demo_human_label_data, demo_metadata_map, probe_human_label_data, probe_metadata_map, all_probe_QIDs, split_probe_human_ids, all_statements_QIDs = load_data(
        demo_mode, probe_mode, probe_setup_id, data_version)

    all_evaluation_data = {"test": [], "val": []}
    for split in tqdm(split_probe_human_ids, desc="Compiling evaluation data"):
        probe_human_ids = split_probe_human_ids[split]
        for probe_human_id in probe_human_ids:
            demo_human_label_d = demo_human_label_data[probe_human_id]
            probe_human_label_d = probe_human_label_data[probe_human_id]

            for probe_qid in tqdm(all_probe_QIDs, desc="Compiling evaluation data"):
                eval_data = {}
                demo_qids = []
                demo_statements = []
                for demo_qid in all_statements_QIDs:
                    if demo_qid == probe_qid:
                        continue
                    demo_gaid = demo_human_label_d[demo_qid]
                    if demo_gaid == -99:
                        continue
                    demo_qids.append(demo_qid)
                demo_qids = random.sample(demo_qids, num_demo)
                for demo_qid in demo_qids:
                    demo_gaid = demo_human_label_d[demo_qid]
                    demo_statement = demo_metadata_map[demo_qid]["converted_statements"][demo_gaid]
                    demo_statements.append(demo_statement)

                eval_data["demo_qids"] = demo_qids
                eval_data["demo_statements"] = demo_statements
                eval_data["probe_qid"] = probe_qid
                eval_data["probe_gaid"] = probe_human_label_d[probe_qid]
                eval_data["probe_statement_options"] = get_probe_statement_options(
                    probe_qid, probe_metadata_map)

                eval_data["prompt"], eval_data["response"] = format_train_data(eval_data["demo_statements"],
                                                                               eval_data["probe_statement_options"],
                                                                               eval_data["probe_gaid"],
                                                                               template_version=template_version)
                eval_data["messages"] = [{"role": "user", "content": eval_data["prompt"]},
                                         {"role": "assistant", "content": eval_data["response"]}]

                all_evaluation_data[split].append(eval_data)

    for split in all_evaluation_data:
        data_save_path = f"data/WVS/training_expts/{split}/{eval_data_version}/{demo_mode[0]}{num_demo}_{probe_mode[0]}1_{data_version}_v{probe_setup_id}.jsonl"
        write_standard_data(all_evaluation_data[split], data_save_path)


def compile_evaluation_data_units_demographics(
        data_version="090824_800",
        template_version=2,
        probe_setup_id=0,
        probe_mode="polar",
        eval_data_version="v2",
):
    demo_human_label_data, demo_metadata_map, probe_human_label_data, probe_metadata_map, all_probe_QIDs, split_probe_human_ids, all_statements_QIDs = load_data(
        "polar", probe_mode, probe_setup_id, data_version)

    all_demographics_QIDs = get_all_demographics_QIDs()

    all_evaluation_data = {"test": [], "val": []}
    for split in tqdm(split_probe_human_ids, desc="Compiling evaluation data"):
        probe_human_ids = split_probe_human_ids[split]
        for probe_human_id in probe_human_ids:
            demo_human_label_d = demo_human_label_data[probe_human_id]
            probe_human_label_d = probe_human_label_data[probe_human_id]

            for probe_qid in tqdm(all_probe_QIDs, desc="Compiling evaluation data"):
                eval_data = {}
                demo_qids = []
                demo_statements = []
                for demo_qid in all_demographics_QIDs:
                    if demo_qid == probe_qid:
                        continue
                    demo_statement = demo_human_label_d[demo_qid]
                    if demo_statement == -99:
                        continue
                    demo_qids.append(demo_qid)
                    demo_statements.append(demo_statement)

                eval_data["demo_qids"] = demo_qids
                eval_data["demo_statements"] = demo_statements
                eval_data["probe_qid"] = probe_qid
                eval_data["probe_gaid"] = probe_human_label_d[probe_qid]
                eval_data["probe_statement_options"] = get_probe_statement_options(
                    probe_qid, probe_metadata_map)

                eval_data["prompt"], eval_data["response"] = format_train_data(eval_data["demo_statements"],
                                                                               eval_data["probe_statement_options"],
                                                                               eval_data["probe_gaid"],
                                                                               template_version=template_version)
                eval_data["messages"] = [{"role": "user", "content": eval_data["prompt"]},
                                         {"role": "assistant", "content": eval_data["response"]}]

                all_evaluation_data[split].append(eval_data)

    for split in all_evaluation_data:
        data_save_path = f"data/WVS/training_expts/{split}/{eval_data_version}/demo_{probe_mode[0]}1_{data_version}_v{probe_setup_id}.jsonl"
        write_standard_data(all_evaluation_data[split], data_save_path)


def compile_human_ids(
        data_version="090824_800",
        template_version=2,
        probe_setup_id=0,
        demo_mode="polar",
        num_demo=200,
        probe_mode="polar",
        eval_data_version="v1",
):
    demo_human_label_data, demo_metadata_map, probe_human_label_data_map, probe_metadata_map, all_probe_QIDs, split_probe_human_ids, all_statements_QIDs = load_data(
        demo_mode, probe_mode, probe_setup_id, data_version)

    all_human_ids = {"test": [], "val": []}
    for split in tqdm(split_probe_human_ids, desc="Compiling evaluation data"):
        probe_human_ids = split_probe_human_ids[split]
        for probe_human_id in probe_human_ids:
            for probe_qid in tqdm(all_probe_QIDs, desc="Compiling evaluation data"):
                all_human_ids[split].append(
                    demo_human_label_data[probe_human_id])

    for split in all_human_ids:
        data_save_path = f"data/WVS/training_expts/{split}/{eval_data_version}/human_ids.jsonl"
        write_standard_data(all_human_ids[split], data_save_path)


def merge_evaluation_data():
    base_test_path = "data/WVS/training_expts/test"
    base_val_path = "data/WVS/training_expts/val"
    base_eval_path = "data/WVS/training_expts/eval"
    
    for data_version in ["v1", "v2"]:
        all_files = os.listdir(f"{base_test_path}/{data_version}")
        all_files = [f for f in all_files if f.endswith(".jsonl")]
        for file in tqdm(all_files, desc="Merging evaluation data"):
            test_data = load_standard_data(f"{base_test_path}/{data_version}/{file}", is_print=False)
            val_data = load_standard_data(f"{base_val_path}/{data_version}/{file}", is_print=False)
            eval_data = test_data + val_data
            # print(len(eval_data))

            write_standard_data(eval_data, f"{base_eval_path}/{data_version}/{file}")


if __name__ == "__main__":
    merge_evaluation_data()

    # for num_demo in [200, 175, 150, 125, 100, 75, 50]:
    #     for probe_setup_id in range(3):
    #         for demo_mode in ["polar", "refined"]:
    #             for probe_mode in ["polar", "refined"]:
    #                 compile_evaluation_data_units(data_version="090824_800",
    #                                               template_version=2,
    #                                               probe_setup_id=probe_setup_id,
    #                                               demo_mode=demo_mode,
    #                                               num_demo=num_demo,
    #                                               probe_mode=probe_mode)

    # for num_demo in [25, 10]:
    #     for probe_setup_id in range(3):
    #         for demo_mode in ["polar", "refined"]:
    #             for probe_mode in ["polar", "refined"]:
    #                 compile_evaluation_data_units(data_version="090824_800",
    #                                               template_version=2,
    #                                               probe_setup_id=probe_setup_id,
    #                                               demo_mode=demo_mode,
    #                                               num_demo=num_demo,
    #                                               probe_mode=probe_mode)

    # for probe_setup_id in range(3):
    #     for probe_mode in ["polar", "refined"]:
    #         compile_evaluation_data_units_demographics(data_version="090824_800",
    #                                                    template_version=2,
    #                                                    probe_setup_id=probe_setup_id,
    #                                                    probe_mode=probe_mode)

    # compile_human_ids(data_version="090824_800",
    #                   template_version=2,
    #                   probe_setup_id=1,
    #                   demo_mode="refined",
    #                   num_demo=200,
    #                   probe_mode="polar")
