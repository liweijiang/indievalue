from src.utils.main_utils import *
from data.WVS.WVS_conversion import *


def select_probe_data(total_select=1000):
    human_data = load_standard_data(
        "data/WVS/human_labels/demographics_in_nl_statements_combined_full_set.jsonl")
    df_data = pd.DataFrame(human_data)

    # Keep the rows with all probe questions
    all_probe_qids = get_all_probe_qids()
    print("Number of probe questions: ", len(all_probe_qids))
    df_data_probes = df_data[all_probe_qids + ["D_INTERVIEW"]]
    df_data_probes = df_data_probes[df_data_probes != -99].dropna()
    df_data = df_data[df_data["D_INTERVIEW"].isin(
        df_data_probes["D_INTERVIEW"].unique())]
    # Keep the rows with at least 239 valid questions (200 training + 39 probe)
    df_data = df_data[df_data["num_valid_questions"] >= 239]
    print("Total number of rows: ", df_data.shape[0])

    df_data_selected = df_data[df_data["B_COUNTRY_to_continent"]
                               == "I am currently in Oceania"]
    num_oceania = df_data_selected.shape[0]

    avg_num_probes_per_continent = int((total_select - num_oceania) / 5) + 1

    for continent in df_data["B_COUNTRY_to_continent"].unique():
        if continent == "I am currently in Oceania":
            continue
        df_data_continent = df_data[df_data["B_COUNTRY_to_continent"] == continent]
        df_data_continent = df_data_continent[df_data_continent["num_valid_demographics"] >= 24]

        # downsample religion with too many samples
        all_religions = df_data_continent["Q289"].unique()
        downsample_religions = ["I belong to no religion or religious denomination",
                                "I belong to the Muslim religion",
                                "I belong to the Buddhist religion",
                                "I belong to the Roman Catholic religion",
                                "I belong to the Protestant religion",
                                "I belong to some other religion or religious denomination",
                                "I belong to some other Christian (Evangelical/Pentecostal/Fee church/etc.) religion",
                                "I belong to the Orthodox (Russian/Greek/etc.) religion"
                                ]
        df_data_continent_downsample_by_religion = df_data_continent[~df_data_continent["Q289"].isin(
            downsample_religions)]

        for r in all_religions:
            if r in downsample_religions:
                df_data_religion = df_data_continent[df_data_continent["Q289"] == r]

                if df_data_religion.shape[0] > 80:
                    df_data_religion_sampled = df_data_religion.sample(80)
                else:
                    df_data_religion_sampled = df_data_religion
                df_data_continent_downsample_by_religion = pd.concat(
                    [df_data_continent_downsample_by_religion, df_data_religion_sampled])

        df_data_continent = df_data_continent_downsample_by_religion

        df_data_sampled = df_data_continent.sample(
            avg_num_probes_per_continent)
        df_data_selected = pd.concat([df_data_selected, df_data_sampled])

    df_data_selected = df_data_selected.sample(total_select)

    show_demographics_distribution(df_data_selected)

    save_path = f"data/WVS/human_labels/probe_data_090524_{total_select}_5.jsonl"
    write_standard_data(df_data_selected.to_dict(orient="records"), save_path)


def get_formatted_probes_questions_individual_question(probe_setup_QIDs, statements_meta_data):
    all_formatted_questions_individual_question = {}
    for qid in probe_setup_QIDs:
        formatted_questions = {}
        for i, qid in enumerate([qid], 1):
            group_name = f"new statement group {i} (NSG{i})"
            formatted_questions[group_name] = []

            converted_statements = statements_meta_data[qid]["converted_statements"]
            for j, statement in enumerate(converted_statements, 1):
                statement_key = f"NSG{i}_s{j}"
                formatted_questions[group_name].append(
                    {statement_key: statement})

        all_formatted_questions_individual_question[qid] = json.dumps(
            formatted_questions, indent=4)

    return all_formatted_questions_individual_question


def compile_probe_prompts_individual_question(num_stmts=200, probe_setup_id=0, mode="stmt", data_version="probe_data_090524_800"):
    expt_id = f"{mode}-{data_version}-v{probe_setup_id}-num_stmt_{num_stmts}-ind_ques"

    human_probe_data = load_standard_data(
        f"data/WVS/human_labels/{data_version}.jsonl")
    statements_meta_data = get_statements_metadata_map()
    prompt_template_path = "data/util_prompts/probe_wvs_statements_v1.txt"
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read()

    all_demographics_QIDs = get_all_demographics_QIDs()
    all_statements_ids = get_all_statements_ids()

    probe_setup_QIDs = get_all_probe_qids(
        probe_setup_id=probe_setup_id)
    formatted_probe_questions_individual_question = get_formatted_probes_questions_individual_question(
        probe_setup_QIDs, statements_meta_data)

    all_data_to_save = []
    for i, d in enumerate(human_probe_data):
        all_statements = []
        all_demographics = []
        if mode == "stmt":
            for qid in d:
                if qid not in probe_setup_QIDs and qid in all_statements_ids:
                    agid = d[qid]
                    if agid == -99:
                        continue
                    converted_statements = statements_meta_data[qid]["converted_statements"]
                    ans_statement = converted_statements[agid]
                    all_statements.append(ans_statement)

        elif mode == "stmt_demographics":
            for qid in d:
                if qid not in probe_setup_QIDs and qid in all_statements_ids:
                    agid = d[qid]
                    if agid == -99:
                        continue
                    converted_statements = statements_meta_data[qid]["converted_statements"]
                    ans_statement = converted_statements[agid]
                    all_statements.append(ans_statement)
                elif qid in all_demographics_QIDs:
                    if d[qid] == -99:
                        continue
                    all_demographics.append(d[qid])

        try:
            all_statements = all_demographics + \
                random.sample(all_statements, num_stmts)
        except:
            all_statements = all_demographics

        formatted_statements = get_formatted_statements(all_statements)

        for probe_qid, formatted_probe_questions in formatted_probe_questions_individual_question.items():
            prompt = prompt_template.replace("{known_statements}", formatted_statements).replace(
                "{probe_questions}", formatted_probe_questions)
            d_dup = d.copy()
            d_dup["probe_prompt"] = prompt
            d_dup["probe_qid"] = probe_qid
            all_data_to_save.append(d_dup)

    save_path = f"data/WVS/probe_expts/{data_version}/{expt_id}.jsonl"
    write_standard_data(all_data_to_save, save_path)


def get_demographics_metadata_map():
    data = load_standard_data(
        "data/WVS/meta_data/demographics_meta_data.jsonl")
    return {d["question_id"]: d for d in data}


def get_statements_metadata_map(is_refined=False):
    if is_refined:
        data = load_standard_data(
            "data/WVS/meta_data/refined_statements_meta_data.jsonl")
    else:
        data = load_standard_data(
            "data/WVS/meta_data/statements_meta_data.jsonl")
    return {d["question_id"]: d for d in data}


def get_formatted_statements(all_statements):
    statements_str = ""
    for s in all_statements:
        statements_str += f"# {s[0].upper()}{s[1:]}\n"
    return statements_str[:-1]


def get_formatted_probes_questions(probe_setup_QIDs, statements_meta_data):
    formatted_questions = {}
    for i, qid in enumerate(probe_setup_QIDs, 1):
        group_name = f"new statement group {i} (NSG{i})"
        formatted_questions[group_name] = []

        converted_statements = statements_meta_data[qid]["converted_statements"]
        for j, statement in enumerate(converted_statements, 1):
            statement_key = f"NSG{i}_s{j}"
            formatted_questions[group_name].append({statement_key: statement})

    return json.dumps(formatted_questions, indent=4)


def show_demographics_distribution(df_data):
    demographics_qids = core_demographics_QIDs
    for qid in demographics_qids:
        print("=" * 50)
        print(df_data[qid].value_counts())


def compile_probe_data_v2(is_refined=False):
    ref_data_path = "data/WVS/human_labels/probe_data/090524_800.jsonl"
    ref_data = load_standard_data(ref_data_path)
    all_interview_ids = [d["D_INTERVIEW"] for d in ref_data]

    refined_string = "_refined" if is_refined else ""
    labeled_data = load_standard_data(
        f"data/WVS/human_labels/demographics_in_nl{refined_string}_statements_combined_full_set.jsonl")
    labeled_data_selected = [
        d for d in labeled_data if d["D_INTERVIEW"] in all_interview_ids]
    labeled_data_selected = sorted(
        labeled_data_selected, key=lambda x: all_interview_ids.index(x["D_INTERVIEW"]))
    save_path = f"data/WVS/human_labels/probe_data/090824_800{refined_string}.jsonl"
    write_standard_data(labeled_data_selected, save_path)


def compile_probe_prompts(num_stmts=200,
                          probe_setup_id=0,
                          mode="stmt",
                          data_version="090824_800",
                          util_prompt_version=1):
    expt_id = f"{mode}-{data_version}-v{probe_setup_id}-num_stmt_{num_stmts}"

    human_probe_data = load_standard_data(
        f"data/WVS/human_labels/probe_data/{data_version}.jsonl")
    if "refined" in data_version:
        statements_meta_data = get_statements_metadata_map(is_refined=True)
    else:
        statements_meta_data = get_statements_metadata_map(is_refined=False)
    prompt_template_path = f"data/util_prompts/probe_wvs_statements_v{util_prompt_version}.txt"
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read()

    all_demographics_QIDs = get_all_demographics_QIDs()
    all_statements_QIDs = get_all_statements_QIDs()

    probe_setup_QIDs = get_all_probe_qids(
        probe_setup_id=probe_setup_id)
    formatted_probe_questions = get_formatted_probes_questions(
        probe_setup_QIDs, statements_meta_data)

    for i, d in enumerate(human_probe_data):
        all_statements_qids = []
        all_demographics = []
        if mode == "stmt":
            for qid in d:
                if qid not in probe_setup_QIDs and qid in all_statements_QIDs:
                    gaid = d[qid]
                    if gaid == -99:
                        continue
                    all_statements_qids.append(qid)

        elif mode == "stmt_demographics":
            for qid in d:
                if qid not in probe_setup_QIDs and qid in all_statements_QIDs:
                    gaid = d[qid]
                    if gaid == -99:
                        continue
                    all_statements_qids.append(qid)

                elif qid in all_demographics_QIDs:
                    if d[qid] == -99:
                        continue
                    all_demographics.append(d[qid])

        if num_stmts != 0:
            all_statements_qids = random.sample(
                all_statements_qids, min(num_stmts, len(all_statements_qids)))
        else:
            all_statements_qids = []

        all_statements = []
        for qid in all_statements_qids:
            if qid in probe_setup_QIDs:
                raise ValueError(f"Qid {qid} is in probe setup QIDs")
            converted_statements = statements_meta_data[qid]["converted_statements"]
            ans_statement = converted_statements[d[qid]]
            all_statements.append(ans_statement)
        all_statements = all_demographics + all_statements

        print("Total statements: ", len(
            all_statements), len(all_statements_qids))
        formatted_statements = get_formatted_statements(all_statements)
        prompt = prompt_template.replace("{known_statements}", formatted_statements).replace(
            "{probe_questions}", formatted_probe_questions)
        d["probe_prompt"] = prompt
        d["selected_statements"] = all_statements_qids

    save_path = f"data/WVS/probe_expts/{data_version}/{expt_id}.jsonl"
    write_standard_data(human_probe_data, save_path)


def double_check_probe_prompts(num_stmts=200,
                               probe_setup_id=0,
                               mode="stmt",
                               data_version="090824_800",
                               util_prompt_version=1):
    expt_id = f"{mode}-{data_version}-v{probe_setup_id}-num_stmt_{num_stmts}"

    save_path = f"data/WVS/probe_expts/{data_version}/{expt_id}.jsonl"
    data = load_standard_data(save_path)

    for d in data:
        print(d["probe_prompt"])
        return


def sample_probe_data(is_refined=False, sample_size=100):
    refined_string = "_refined" if is_refined else ""
    data_path = f"data/WVS/human_labels/probe_data/090824_800{refined_string}.jsonl"
    data = load_standard_data(data_path)
    data_selected = random.sample(data, sample_size)
    save_path = data_path.replace(".jsonl", f"_sample_{sample_size}.jsonl")
    write_standard_data(data_selected, save_path)


def compile_probe_data_by_category(data_version="090824_800_sample_100",
                                   probe_setup_id=0,
                                   mode="stmt",
                                   util_prompt_version=1):
    data = load_standard_data(
        f"data/WVS/human_labels/probe_data/{data_version}.jsonl")
    all_demographics_QIDs = get_all_demographics_QIDs()
    probe_setup_QIDs_by_category = get_probe_setups()[
        probe_setup_id]
    all_probe_setup_QIDs = get_all_probe_qids(
        probe_setup_id=probe_setup_id)
    statements_QIDs_by_category = get_statements_qids_by_category()

    prompt_template_path = f"data/util_prompts/probe_wvs_statements_v{util_prompt_version}.txt"
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read()

    human_probe_data = load_standard_data(
        f"data/WVS/human_labels/probe_data/{data_version}.jsonl")
    if "refined" in data_version:
        statements_meta_data = get_statements_metadata_map(is_refined=True)
    else:
        statements_meta_data = get_statements_metadata_map(is_refined=False)

    formatted_probe_questions = get_formatted_probes_questions(
        all_probe_setup_QIDs, statements_meta_data)

    for category in probe_setup_QIDs_by_category:
        category_id = category.lower().replace(" ", "_")
        expt_id = f"{mode}-{data_version}-v{probe_setup_id}-num_stmt_{category_id}"

        for i, d in enumerate(human_probe_data):
            all_statements_qids = []
            all_demographics = []
            if mode == "stmt":
                for qid in d:
                    if qid in statements_QIDs_by_category[category] and qid not in all_probe_setup_QIDs:
                        gaid = d[qid]
                        if gaid == -99:
                            continue
                        all_statements_qids.append(qid)

            elif mode == "stmt_demographics":
                for qid in d:
                    if qid in statements_QIDs_by_category[category] and qid not in all_probe_setup_QIDs:
                        gaid = d[qid]
                        if gaid == -99:
                            continue
                        all_statements_qids.append(qid)

                    elif qid in all_demographics_QIDs:
                        if d[qid] == -99:
                            continue
                        all_demographics.append(d[qid])

            all_statements = []
            for qid in all_statements_qids:
                if qid in all_probe_setup_QIDs:
                    raise ValueError(f"Qid {qid} is in probe setup QIDs")
                converted_statements = statements_meta_data[qid]["converted_statements"]
                ans_statement = converted_statements[d[qid]]
                all_statements.append(ans_statement)
            all_statements = all_demographics + all_statements

            print("Total statements: ", len(
                all_statements), len(all_statements_qids))
            formatted_statements = get_formatted_statements(all_statements)
            prompt = prompt_template.replace("{known_statements}", formatted_statements).replace(
                "{probe_questions}", formatted_probe_questions)
            d["probe_prompt"] = prompt
            d["selected_statements"] = all_statements_qids

        save_path = f"data/WVS/probe_expts/{data_version}/{expt_id}.jsonl"
        write_standard_data(human_probe_data, save_path)


def compile_probe_prompts_refined(num_stmts=200,
                                  probe_setup_id=0,
                                  mode="stmt",
                                  data_version="090824_800_refined",
                                  util_prompt_version=1):
    expt_id = f"{mode}-{data_version}-v{probe_setup_id}-num_stmt_{num_stmts}"

    human_probe_data = load_standard_data(
        f"data/WVS/human_labels/probe_data/{data_version}.jsonl")
    statements_meta_data = get_statements_metadata_map(is_refined=True)
    statements_meta_data_polar = get_statements_metadata_map(is_refined=False)
    prompt_template_path = f"data/util_prompts/probe_wvs_statements_v{util_prompt_version}.txt"
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read()

    all_demographics_QIDs = get_all_demographics_QIDs()
    all_statements_QIDs = get_all_statements_QIDs()

    probe_setup_QIDs = get_all_probe_qids(
        probe_setup_id=probe_setup_id)
    formatted_probe_questions = get_formatted_probes_questions(
        probe_setup_QIDs, statements_meta_data_polar)

    for i, d in enumerate(human_probe_data):
        all_statements_qids = []
        all_demographics = []
        if mode == "stmt":
            for qid in d:
                if qid not in probe_setup_QIDs and qid in all_statements_QIDs:
                    gaid = d[qid]
                    if gaid == -99:
                        continue
                    all_statements_qids.append(qid)

        elif mode == "stmt_demographics":
            for qid in d:
                if qid not in probe_setup_QIDs and qid in all_statements_QIDs:
                    gaid = d[qid]
                    if gaid == -99:
                        continue
                    all_statements_qids.append(qid)

                elif qid in all_demographics_QIDs:
                    if d[qid] == -99:
                        continue
                    all_demographics.append(d[qid])

        if num_stmts != 0:
            all_statements_qids = random.sample(
                all_statements_qids, min(num_stmts, len(all_statements_qids)))
        else:
            all_statements_qids = []

        all_statements = []
        for qid in all_statements_qids:
            if qid in probe_setup_QIDs:
                raise ValueError(f"Qid {qid} is in probe setup QIDs")
            converted_statements = statements_meta_data[qid]["converted_statements"]
            ans_statement = converted_statements[d[qid]]
            all_statements.append(ans_statement)
        all_statements = all_demographics + all_statements

        print("Total statements: ", len(
            all_statements), len(all_statements_qids))
        formatted_statements = get_formatted_statements(all_statements)
        prompt = prompt_template.replace("{known_statements}", formatted_statements).replace(
            "{probe_questions}", formatted_probe_questions)
        d["probe_prompt"] = prompt
        d["selected_statements"] = all_statements_qids

    save_path = f"data/WVS/probe_expts/{data_version}/{expt_id}.jsonl"
    write_standard_data(human_probe_data, save_path)


if __name__ == "__main__":
    # data_version = "090824_800"
    # for probe_setup_id in [0, 1, 2]:
    #     for num_stmts in [0, 50, 100, 150, 200]:
    #         compile_probe_prompts(
    #             mode="stmt",
    #             num_stmts=num_stmts,
    #             probe_setup_id=probe_setup_id,
    #             data_version=data_version
    #         )
    #         compile_probe_prompts(
    #             mode="stmt_demographics",
    #             num_stmts=num_stmts,
    #             probe_setup_id=probe_setup_id,
    #             data_version=data_version
    #         )

    # compile_probe_data_v2(is_refined=False)
    # compile_probe_data_v2(is_refined=True)

    data_version = "090824_800_refined"
    for probe_setup_id in [0, 1, 2]:
        for num_stmts in [200]:
            compile_probe_prompts_refined(
                data_version=data_version,
                probe_setup_id=probe_setup_id,
                mode="stmt",
                util_prompt_version=1
            )
