from src.utils.main_utils import *
from data.WVS.WVS_conversion import *
from transformers import AutoTokenizer

data_version = "v4"
split = "train"
unit_save_path = f"data/WVS/training_expts/{split}/{data_version}/units/"
mixture_save_path = f"data/WVS/training_expts/{split}/{data_version}/mixtures/"


def look_at_data_units(num_humans_per_question=200,
                       demo_mode="polar",
                       num_demo=200,
                       probe_mode="polar",
                       question_set="full",
                       template_version=2):
    train_data_save_path = unit_save_path + \
        f"{demo_mode[0]}{num_demo}_{probe_mode[0]}1_n{num_humans_per_question}_{question_set}.jsonl"
    train_data = load_standard_data(train_data_save_path)

    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct",
    #                                           trust_remote_code=True)

    # formatted_prompts = []
    # formatted_prompts_lens = []
    # for d in tqdm(train_data, desc="Looking at data units"):
    #     prompt, response = format_train_data(demo_statements=d["demo_statements"],
    #                                          probe_statement_options=d["probe_statement_options"],
    #                                          probe_gaid=d["probe_gaid"],
    #                                          template_version=2)
    #     messages = [{"role": "user", "content": prompt},
    #                 {"role": "assistant", "content": response}]
    #     formatted_m = tokenizer.apply_chat_template(messages,
    #                                                 tokenize=True,
    #                                                 add_generation_prompt=True)
    #     formatted_prompts.append(formatted_m)
    #     formatted_prompts_lens.append(len(formatted_m))

    # print(f"Min prompt length: {min(formatted_prompts_lens)}")
    # print(f"Max prompt length: {max(formatted_prompts_lens)}")
    # print(
    #     f"Mean prompt length: {sum(formatted_prompts_lens) / len(formatted_prompts_lens)}")


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

    # probe_statement_dict = {}
    # for i, probe_statement in enumerate(probe_statement_options):
    #     probe_statement_dict[f"Option {i+1}"] = probe_statement
    # probe_statement_string = json.dumps(probe_statement_dict, indent=4)

    probe_statement_string = ""
    for i, probe_statement in enumerate(probe_statement_options):
        probe_statement_string += f"Option {i+1}: {probe_statement}\n"

    response = f"Option {probe_gaid + 1}: {probe_statement_options[probe_gaid]}"

    prompt = prompt_template.replace(
        "{known_statements}", demo_statements_string)
    prompt = prompt.replace("{probe_statements}", probe_statement_string)
    return prompt, response


def sample_demo_human_label_data(demo_human_label_data,
                                 candidate_QIDs,
                                 num_humans_per_question,
                                 num_demo,
                                 all_statements_QIDs):
    num_demo_mode = num_demo
    sampled_demo_human_label_data = {probe_qid: []
                                     for probe_qid in candidate_QIDs}
    for probe_qid in tqdm(candidate_QIDs, desc="Sampling demo human label data"):
        random.shuffle(demo_human_label_data)
        valid_human_ids = []
        for human_label_d in demo_human_label_data:
            if num_demo_mode == "rand":
                num_demo = random.randint(100, 200)
            elif num_demo_mode == "mixed":
                num_demo = random.randint(50, 200)

            num_valid_questions = human_label_d["num_valid_questions"]
            gaid = human_label_d[probe_qid]
            human_id = human_label_d["D_INTERVIEW"]
            if human_id not in valid_human_ids and gaid != -99 and num_valid_questions >= (num_demo + 1):
                valid_human_ids.append(human_id)
                valid_demo_qids = [demo_qid for demo_qid in all_statements_QIDs if demo_qid !=
                                   probe_qid and human_label_d[demo_qid] != -99]
                valid_demo_qids = random.sample(valid_demo_qids, num_demo)
                human_label_d["demo_qids"] = valid_demo_qids
                sampled_demo_human_label_data[probe_qid].append(human_label_d)

                if len(sampled_demo_human_label_data[probe_qid]) == num_humans_per_question:
                    break
        if len(sampled_demo_human_label_data[probe_qid]) < num_humans_per_question:
            raise Exception(f"Not enough data for {probe_qid}")
    return sampled_demo_human_label_data


def get_probe_statement_options(probe_qid, probe_metadata_map):
    return probe_metadata_map[probe_qid]["converted_statements"]


def get_human_id_to_human_data_map(human_label_data):
    human_id_to_human_data_map = {}
    for human_label_d in human_label_data:
        human_id = human_label_d["D_INTERVIEW"]
        human_id_to_human_data_map[human_id] = human_label_d
    return human_id_to_human_data_map


def load_probe_data_human_ids():
    data_path = "data/WVS/human_labels/probe_data/090824_800.jsonl"
    probe_data = load_standard_data(data_path, is_print=False)
    return [data_d["D_INTERVIEW"] for data_d in probe_data]


def load_data(demo_mode, probe_mode, question_set, continent_name=None):
    probe_human_ids = load_probe_data_human_ids()
    refined_human_label_data = get_human_label_data(is_refined=True)
    refined_human_label_data = [
        human_label_d for human_label_d in refined_human_label_data if human_label_d["D_INTERVIEW"] not in probe_human_ids]
    polar_human_label_data = get_human_label_data(is_refined=False)
    polar_human_label_data = [
        human_label_d for human_label_d in polar_human_label_data if human_label_d["D_INTERVIEW"] not in probe_human_ids]
    refined_statements_metadata_map = get_statements_metadata_map(
        is_refined=True)
    polar_statements_metadata_map = get_statements_metadata_map(
        is_refined=False)
    if question_set == "full":
        candidate_QIDs = get_all_statements_QIDs()
    elif question_set == "probe":
        candidate_QIDs = get_all_probe_qids()

    all_statements_QIDs = get_all_statements_QIDs()

    if demo_mode == "polar":
        demo_human_label_data = polar_human_label_data
        demo_metadata_map = polar_statements_metadata_map
    else:
        demo_human_label_data = refined_human_label_data
        demo_metadata_map = refined_statements_metadata_map
    if probe_mode == "polar":
        probe_human_label_data_map = get_human_id_to_human_data_map(
            polar_human_label_data)
        probe_metadata_map = polar_statements_metadata_map
    else:
        probe_human_label_data_map = get_human_id_to_human_data_map(
            refined_human_label_data)
        probe_metadata_map = refined_statements_metadata_map

    if continent_name is not None:
        demo_human_label_data = [d for d in demo_human_label_data if continent_name.lower(
        ) in d["B_COUNTRY_to_continent"].lower()]
        probe_human_label_data_map = {k: v for k, v in probe_human_label_data_map.items(
        ) if continent_name.lower() in v["B_COUNTRY_to_continent"].lower()}

    return demo_human_label_data, demo_metadata_map, probe_human_label_data_map, probe_metadata_map, candidate_QIDs, all_statements_QIDs


def compile_data_units(num_humans_per_question=200,
                       demo_mode="polar",
                       num_demo=200,
                       probe_mode="polar",
                       question_set="full",
                       template_version=2):
    demo_human_label_data, demo_metadata_map, probe_human_label_data_map, probe_metadata_map, candidate_QIDs, all_statements_QIDs = load_data(
        demo_mode,
        probe_mode,
        question_set)

    selected_demo_human_label_data = sample_demo_human_label_data(
        demo_human_label_data,
        candidate_QIDs,
        num_humans_per_question,
        num_demo,
        all_statements_QIDs)

    all_train_data = []
    for probe_qid in tqdm(candidate_QIDs, desc="Compiling training data"):
        probe_statement_options = get_probe_statement_options(
            probe_qid, probe_metadata_map)

        for demo_human_label_d in selected_demo_human_label_data[probe_qid]:
            demo_human_label_d["probe_qid"] = probe_qid
            demo_human_label_d["demo_statements"] = []
            for demo_qid in demo_human_label_d["demo_qids"]:
                demo_gaid = demo_human_label_d[demo_qid]
                demo_statement = demo_metadata_map[demo_qid]["converted_statements"][demo_gaid]
                demo_human_label_d["demo_statements"].append(demo_statement)
            demo_human_label_d["probe_statement_options"] = probe_statement_options
            demo_human_label_d["probe_gaid"] = probe_human_label_data_map[demo_human_label_d["D_INTERVIEW"]][probe_qid]

            demo_human_label_d["prompt"], demo_human_label_d["response"] = format_train_data(demo_human_label_d["demo_statements"],
                                                                                             probe_statement_options,
                                                                                             demo_human_label_d["probe_gaid"],
                                                                                             template_version=template_version)
            demo_human_label_d["messages"] = [{"role": "user", "content": demo_human_label_d["prompt"]},
                                              {"role": "assistant", "content": demo_human_label_d["response"]}]

            all_train_data.append(demo_human_label_d)

    train_data_save_path = unit_save_path + \
        f"{demo_mode[0]}{num_demo}_{probe_mode[0]}1_n{num_humans_per_question}_{question_set}.jsonl"
    write_standard_data(all_train_data, train_data_save_path)

    return all_train_data


def compile_data_units_by_continent(num_humans_per_question=200,
                                    demo_mode="polar",
                                    num_demo=200,
                                    probe_mode="polar",
                                    question_set="full",
                                    template_version=2,
                                    continent_name=None):
    demo_human_label_data, demo_metadata_map, probe_human_label_data_map, probe_metadata_map, candidate_QIDs, all_statements_QIDs = load_data(
        demo_mode,
        probe_mode,
        question_set,
        continent_name=continent_name)

    selected_demo_human_label_data = sample_demo_human_label_data(
        demo_human_label_data,
        candidate_QIDs,
        num_humans_per_question,
        num_demo,
        all_statements_QIDs)

    for qid in selected_demo_human_label_data:
        if len(selected_demo_human_label_data[qid]) < num_humans_per_question:
            raise Exception(f"Not enough data for {qid}")
        else:
            print(
                f"Enough data for {qid}: {len(selected_demo_human_label_data[qid])} humans")

    all_train_data = []
    for probe_qid in tqdm(candidate_QIDs, desc="Compiling training data"):
        probe_statement_options = get_probe_statement_options(
            probe_qid, probe_metadata_map)

        for demo_human_label_d in selected_demo_human_label_data[probe_qid]:
            demo_human_label_d["probe_qid"] = probe_qid
            demo_human_label_d["demo_statements"] = []
            for demo_qid in demo_human_label_d["demo_qids"]:
                demo_gaid = demo_human_label_d[demo_qid]
                demo_statement = demo_metadata_map[demo_qid]["converted_statements"][demo_gaid]
                demo_human_label_d["demo_statements"].append(demo_statement)
            demo_human_label_d["probe_statement_options"] = probe_statement_options
            demo_human_label_d["probe_gaid"] = probe_human_label_data_map[demo_human_label_d["D_INTERVIEW"]][probe_qid]

            demo_human_label_d["prompt"], demo_human_label_d["response"] = format_train_data(demo_human_label_d["demo_statements"],
                                                                                             probe_statement_options,
                                                                                             demo_human_label_d["probe_gaid"],
                                                                                             template_version=template_version)
            demo_human_label_d["messages"] = [{"role": "user", "content": demo_human_label_d["prompt"]},
                                              {"role": "assistant", "content": demo_human_label_d["response"]}]

            all_train_data.append(demo_human_label_d)

    train_data_save_path = unit_save_path + \
        f"{demo_mode[0]}{num_demo}_{probe_mode[0]}1_n{num_humans_per_question}_{question_set}_{continent_name}.jsonl"
    write_standard_data(all_train_data, train_data_save_path)

    return all_train_data


def sample_demo_human_label_data_demographics(demo_human_label_data,
                                              candidate_QIDs,
                                              num_humans_per_question,
                                              all_demographics_QIDs):
    sampled_demo_human_label_data = {probe_qid: []
                                     for probe_qid in candidate_QIDs}
    for probe_qid in tqdm(candidate_QIDs, desc="Sampling demo human label data"):
        random.shuffle(demo_human_label_data)
        valid_human_ids = []
        for human_label_d in demo_human_label_data:
            gaid = human_label_d[probe_qid]
            human_id = human_label_d["D_INTERVIEW"]
            if human_id not in valid_human_ids and gaid != -99:
                valid_demo_qids = [demo_qid for demo_qid in all_demographics_QIDs if demo_qid !=
                                   probe_qid and human_label_d[demo_qid] != -99]
                valid_human_ids.append(human_id)
                human_label_d["demo_qids"] = valid_demo_qids
                sampled_demo_human_label_data[probe_qid].append(human_label_d)

                if len(sampled_demo_human_label_data[probe_qid]) == num_humans_per_question:
                    break
        if len(sampled_demo_human_label_data[probe_qid]) < num_humans_per_question:
            raise Exception(f"Not enough data for {probe_qid}")
    return sampled_demo_human_label_data


def compile_data_units_demographics(num_humans_per_question=200,
                                    demo_mode="polar",
                                    probe_mode="polar",
                                    question_set="full",
                                    template_version=2):
    demo_human_label_data, demo_metadata_map, probe_human_label_data_map, probe_metadata_map, candidate_QIDs, all_statements_QIDs = load_data(
        demo_mode,
        probe_mode,
        question_set)

    all_demographics_QIDs = get_all_demographics_QIDs()

    selected_demo_human_label_data = sample_demo_human_label_data_demographics(
        demo_human_label_data,
        candidate_QIDs,
        num_humans_per_question,
        all_demographics_QIDs)

    all_train_data = []
    for probe_qid in tqdm(candidate_QIDs, desc="Compiling training data"):
        probe_statement_options = get_probe_statement_options(
            probe_qid, probe_metadata_map)

        for demo_human_label_d in selected_demo_human_label_data[probe_qid]:
            demo_human_label_d["probe_qid"] = probe_qid
            demo_human_label_d["demo_statements"] = []
            for demo_qid in demo_human_label_d["demo_qids"]:
                demo_statement = demo_human_label_d[demo_qid]
                demo_human_label_d["demo_statements"].append(demo_statement)
            demo_human_label_d["probe_statement_options"] = probe_statement_options
            demo_human_label_d["probe_gaid"] = probe_human_label_data_map[demo_human_label_d["D_INTERVIEW"]][probe_qid]

            demo_human_label_d["prompt"], demo_human_label_d["response"] = format_train_data(demo_human_label_d["demo_statements"],
                                                                                             probe_statement_options,
                                                                                             demo_human_label_d["probe_gaid"],
                                                                                             template_version=template_version)
            demo_human_label_d["messages"] = [{"role": "user", "content": demo_human_label_d["prompt"]},
                                              {"role": "assistant", "content": demo_human_label_d["response"]}]
            all_train_data.append(demo_human_label_d)

    train_data_save_path = unit_save_path + \
        f"demo_{probe_mode[0]}1_n{num_humans_per_question}_{question_set}.jsonl"
    write_standard_data(all_train_data, train_data_save_path)

    return all_train_data


if __name__ == "__main__":
    template_version = 2

    # for continent_name in continent_name_to_country_name.keys():
    #     for num_humans_per_question in [50, 100, 150, 200]:
    #         for demo_mode in ["polar", "refined"]:
    #             for probe_mode in ["polar", "refined"]:
    #                 for question_set in ["full"]:
    #                     compile_data_units_by_continent(num_humans_per_question=num_humans_per_question,
    #                                                     demo_mode=demo_mode,
    #                                                     num_demo=200,
    #                                                     probe_mode=probe_mode,
    #                                                     question_set=question_set,
    #                                                     template_version=template_version,
    #                                                     continent_name=continent_name)

    # for num_humans_per_question in [400, 600, 800, 1000]:
    #     for demo_mode in ["polar", "refined"]:
    #         for probe_mode in ["polar", "refined"]:
    #             for question_set in ["full"]:
    #                 compile_data_units(num_humans_per_question=num_humans_per_question,
    #                                    demo_mode=demo_mode,
    #                                    num_demo=200,
    #                                    probe_mode=probe_mode,
    #                                    question_set=question_set,
    #                                    template_version=template_version)

    # for num_humans_per_question in [175, 125, 75, 25]:
    #     for demo_mode in ["polar", "refined"]:
    #         for probe_mode in ["polar", "refined"]:
    #             for question_set in ["full"]:
    #                 compile_data_units(num_humans_per_question=num_humans_per_question,
    #                                    demo_mode=demo_mode,
    #                                    num_demo=200,
    #                                    probe_mode=probe_mode,
    #                                    question_set=question_set,
    #                                    template_version=template_version)

    # for num_humans_per_question in [200, 150, 100, 50]:
    #     for demo_mode in ["polar", "refined"]:
    #         for probe_mode in ["polar", "refined"]:
    #             for question_set in ["full"]:
    #                 look_at_data_units(num_humans_per_question=num_humans_per_question,
    #                                    demo_mode=demo_mode,
    #                                    num_demo="rand",
    #                                    probe_mode=probe_mode,
    #                                    question_set=question_set,
    #                                    template_version=template_version)

    # num_demo = 200
    # num_humans_per_question = 200
    # demo_mode = "refined"
    # probe_mode = "binary"
    # question_set = "probe"

    # train_data_save_path = unit_save_path + \
    #     f"{demo_mode[0]}{num_demo}_{probe_mode[0]}1_n{num_humans_per_question}_{question_set}.jsonl"
    # look_at_data_units(num_humans_per_question=num_humans_per_question,
    #                    demo_mode=demo_mode,
    #                    num_demo=num_demo,
    #                    probe_mode=probe_mode,
    #                    question_set=question_set)

    # for num_humans_per_question in [100, 200, 400, 800]:
    #     for demo_mode in ["polar"]:
    #         for probe_mode in ["polar", "refined"]:
    #             compile_data_units_demographics(num_humans_per_question=num_humans_per_question,
    #                                             demo_mode=demo_mode,
    #                                             probe_mode=probe_mode,
    #                                             question_set="full",
    #                                             template_version=template_version)

    # for num_demo in ["mixed"]:
    #     for num_humans_per_question in [50, 100, 150, 200, 250, 300]:
    #         for demo_mode in ["polar", "refined"]:
    #             for probe_mode in ["polar", "refined"]:
    #                 for question_set in ["full"]:
    #                     compile_data_units(num_humans_per_question=num_humans_per_question,
    #                                        demo_mode=demo_mode,
    #                                        num_demo=num_demo,
    #                                        probe_mode=probe_mode,
    #                                        question_set=question_set,
    #                                        template_version=template_version)


    # for num_demo in ["mixed", 200]:
    #     for num_humans_per_question in [25, 75]:
    #         for demo_mode in ["polar", "refined"]:
    #             for probe_mode in ["polar", "refined"]:
    #                 for question_set in ["full"]:
    #                     compile_data_units(num_humans_per_question=num_humans_per_question,
    #                                        demo_mode=demo_mode,
    #                                        num_demo=num_demo,
    #                                        probe_mode=probe_mode,
    #                                        question_set=question_set,
    #                                        template_version=template_version)


    for num_demo in ["mixed", 200]:
        for num_humans_per_question in [300, 400, 500]:
            for demo_mode in ["polar", "refined"]:
                for probe_mode in ["polar", "refined"]:
                    for question_set in ["full"]:
                        compile_data_units(num_humans_per_question=num_humans_per_question,
                                           demo_mode=demo_mode,
                                           num_demo=num_demo,
                                           probe_mode=probe_mode,
                                           question_set=question_set,
                                           template_version=template_version)
