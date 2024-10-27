import data.WVS.WVS_conversion as wvs_conversion_data
import pandas as pd
from src.utils.main_utils import *
from tqdm import tqdm
import numpy as np


def add_converted_data(d, q, qid, answer_vars, answer_ids_vars, template_statement, is_refined, question_type, question_category):
    d_converted_data = {}
    d_converted_data["question_id"] = qid
    d_converted_data["original_question"] = d["original_question"]
    d_converted_data["original_question_supplement"] = d["original_question_supplement"] if "original_question_supplement" in d else ""
    d_converted_data["answer_ids"] = answer_ids_vars
    d_converted_data["all_answer_ids"] = []
    for aids in answer_ids_vars:
        d_converted_data["all_answer_ids"] += aids
    d_converted_data["grouped_answer_ids"] = [
        i for i in range(len(answer_ids_vars))]
    d_converted_data["grouped_answer_ids_to_answer_ids"] = {
        i: answer_ids_vars[i] for i in range(len(answer_ids_vars))}
    d_converted_data["answer_ids_to_grouped_answer_ids"] = {}
    for i, aids in enumerate(answer_ids_vars):
        for aid in aids:
            d_converted_data["answer_ids_to_grouped_answer_ids"][aid] = i
    d_converted_data["converted_statements"] = []
    d_converted_data["answer_vars"] = answer_vars
    d_converted_data["question_vars"] = q
    d_converted_data["statement_template"] = template_statement
    d_converted_data["is_refined"] = is_refined
    d_converted_data["question_type"] = question_type
    d_converted_data["question_category"] = question_category
    return d_converted_data


def compile_statements_data(is_print=False,
                            is_refined=False,
                            is_save=True):
    all_statements_data = wvs_conversion_data.statements_data

    all_data = []
    for d in all_statements_data:
        question_type = d["question_type"][0]
        question_category = d["question_category"]
        is_refined_exists = "refined_variable_names" in d

        variable_names = d["variable_names"]
        answer_ids_var_name = "answer_ids"
        template_statement = d["conversion_template"][0]
        if is_refined and is_refined_exists:
            variable_names = d["refined_variable_names"]
            answer_ids_var_name = "refined_answer_ids"
            template_statement = d["refined_conversion_template"][0]

        if len(d["question_ids"]) > 1:
            if "answer" in variable_names[0]:
                answer_var_name = variable_names[0]
                question_var_name = variable_names[1]
            else:
                question_var_name = variable_names[0]
                answer_var_name = variable_names[1]
            answer_ids_vars = d[answer_ids_var_name]
            answer_vars = d[answer_var_name]
            question_vars = d[question_var_name]

            template_statement_clean = template_statement.replace(
                f"{question_var_name}", "question_var").replace(f"{answer_var_name}", "answer_var")

            # ==================== check ill-formed data ====================
            if len(d["question_ids"]) != len(question_vars):
                print(d["question_ids"])
                print(question_vars)
                raise ValueError(
                    "question_ids and question_vars have different lengths")

            if len(d[answer_ids_var_name]) != len(answer_vars):
                print(d[answer_ids_var_name])
                print(answer_vars)
                raise ValueError(
                    "answer_ids and answer_vars have different lengths")
            # ==================== check ill-formed data ====================

            for qid, q in zip(d["question_ids"], question_vars):
                d_converted_data = add_converted_data(d,
                                                      q,
                                                      qid,
                                                      answer_vars,
                                                      answer_ids_vars,
                                                      template_statement_clean,
                                                      is_refined,
                                                      question_type,
                                                      question_category)

                for aid, a in enumerate(answer_vars):
                    filled_template = template_statement.replace(
                        "{" + question_var_name + "}", q)
                    filled_template = filled_template.replace(
                        "{" + answer_var_name + "}", a)
                    if is_print:
                        print(qid, aid, filled_template)
                    d_converted_data["converted_statements"].append(
                        filled_template)
                all_data.append(d_converted_data)

                if is_print:
                    pretty_print_dict(d_converted_data)

        elif len(d["question_ids"]) == 1:
            qid = d["question_ids"][0]
            answer_var_name = variable_names[0]
            answer_vars = d[answer_var_name]
            answer_ids_vars = d[answer_ids_var_name]

            template_statement_clean = template_statement.replace(
                f"{answer_var_name}", "answer_var")

            d_converted_data = add_converted_data(d,
                                                  "",
                                                  qid,
                                                  answer_vars,
                                                  answer_ids_vars,
                                                  template_statement_clean,
                                                  is_refined,
                                                  question_type,
                                                  question_category)

            for aid, a in enumerate(answer_vars):
                filled_template = template_statement.replace(
                    "{" + answer_var_name + "}", a)
                d_converted_data["converted_statements"].append(
                    filled_template)
            all_data.append(d_converted_data)

            if is_print:
                pretty_print_dict(d_converted_data)

    if is_save:
        if is_refined:
            save_path = "data/WVS/meta_data/refined_statements_meta_data.jsonl"
        else:
            save_path = "data/WVS/meta_data/statements_meta_data.jsonl"
        write_standard_data(all_data, save_path)


def look_at_statements_data(is_refined=False, is_skip_done=False, is_print=False):
    if is_refined:
        save_path = "data/WVS/WVS_wave_7_refined_converted.jsonl"
    else:
        save_path = "data/WVS/WVS_wave_7_converted.jsonl"
    data = load_standard_data(save_path)

    done_qids = [
        "Q259", "Q258", "Q257", "Q256", "Q255", "Q254", "Q253", "Q252",
        "Q251", "Q250", "Q249", "Q248", "Q247", "Q246", "Q245", "Q244",
        "Q243", "Q242", "Q241", "Q240", "Q239", "Q238", "Q237", "Q236",
        "Q235", "Q234", "Q233", "Q232", "Q231", "Q230", "Q229", "Q228",
        "Q227", "Q226", "Q225", "Q224", "Q222", "Q221", "Q220", "Q219",
        "Q218", "Q217", "Q216", "Q215", "Q214", "Q213", "Q212", "Q211",
        "Q210", "Q209", "Q208", "Q207", "Q206", "Q205", "Q204", "Q203",
        "Q202", "Q201", "Q200", "Q199", "Q198", "Q197", "Q196", "Q195",
        "Q194", "Q193", "Q192", "Q191", "Q190", "Q189", "Q188", "Q187",
        "Q186", "Q185", "Q184", "Q183", "Q182", "Q181", "Q180", "Q179",
        "Q178", "Q177", "Q176", "Q175", "Q174", "Q173", "Q172", "Q171",
        "Q170", "Q169", "Q168", "Q167", "Q166", "Q165", "Q164", "Q163",
        "Q162", "Q161", "Q160", "Q159", "Q158", "Q157", "Q156", "Q155",
        "Q154", "Q153", "Q152", "Q151", "Q150", "Q149", "Q148", "Q147",
        "Q146", "Q145", "Q144", "Q143", "Q142", "Q141", "Q140", "Q139",
        "Q138", "Q137", "Q136", "Q135", "Q134", "Q133", "Q132", "Q131",
        "Q130", "Q129", "Q128", "Q127", "Q126", "Q125", "Q124", "Q123",
        "Q122", "Q121", "Q120", "Q119", "Q118", "Q117", "Q116", "Q115",
        "Q114", "Q113", "Q112", "Q111", "Q110", "Q109", "Q108", "Q107",
        "Q106", "Q104", "Q103", "Q102", "Q101", "Q100", "Q99",
        "Q98", "Q97", "Q96", "Q95", "Q94", "Q90", "Q89", "Q88", "Q87",
        "Q86", "Q85", "Q84", "Q83", "Q81", "Q80", "Q79", "Q78", "Q77",
        "Q76", "Q75", "Q74", "Q73", "Q72", "Q71", "Q70", "Q69", "Q68",
        "Q67", "Q66", "Q65", "Q64", "Q63", "Q62", "Q61", "Q60", "Q59",
        "Q58", "Q57", "Q56", "Q55", "Q54", "Q53", "Q52", "Q51", "Q50",
        "Q49", "Q48", "Q47", "Q46", "Q45", "Q44", "Q43", "Q42", "Q41",
        "Q40", "Q39", "Q38", "Q37", "Q36", "Q35", "Q34", "Q33", "Q32",
        "Q31", "Q30", "Q29", "Q28", "Q27", "Q26", "Q25", "Q24", "Q23",
        "Q22", "Q21", "Q20", "Q19", "Q18", "Q17", "Q16", "Q15", "Q14",
        "Q13", "Q12", "Q11", "Q10", "Q9", "Q8", "Q7", "Q6", "Q5", "Q4",
        "Q3", "Q2", "Q1"]

    combo_count = 1
    group_data_by_category = {}
    all_converted_statements = []
    for d in data:
        qid = d["question_id"]
        if is_skip_done and qid in done_qids:
            continue

        if is_print:
            print("=" * 100)
            pretty_print_dict(d)

        converted_statements = d["converted_statements"]
        all_converted_statements += converted_statements

        combo_count *= len(converted_statements)

        question_category = d["question_category"]
        if question_category not in group_data_by_category:
            group_data_by_category[question_category] = {
                "questions": [], "statements": []}
        group_data_by_category[question_category]["questions"].append(d)
        group_data_by_category[question_category]["statements"] += converted_statements

    print(len(all_converted_statements))
    print("Combo count: ", "{:e}".format(combo_count))

    for question_category, data in group_data_by_category.items():
        print("=" * 100)
        print(question_category)
        print("Number of questions: ", len(data["questions"]))
        print("Number of statements: ", len(data["statements"]))
        print("Avg. number of statements per question: ", "{:.2f}".format(
            len(data["statements"]) / len(data["questions"])))
    print("=" * 100)
    print("Total number of question categories: ", len(group_data_by_category))
    print("Total number of questions: ", sum(
        [len(data["questions"]) for data in group_data_by_category.values()]))
    print("Total number of statements: ", sum(
        [len(data["statements"]) for data in group_data_by_category.values()]))


def load_statements_data(is_refined=False):
    if is_refined:
        data_path = "data/WVS/meta_data/refined_statements_meta_data.jsonl"
    else:
        data_path = "data/WVS/meta_data/statements_meta_data.jsonl"
    wvs_statements_data = load_standard_data(data_path)
    all_QIDs = [d["question_id"] for d in wvs_statements_data]

    wvs_statements_data_map = {}
    qid_to_converted_statements_map = {}
    for d in wvs_statements_data:
        qid = d["question_id"]
        converted_statements = d["converted_statements"]
        wvs_statements_data_map[qid] = d
        qid_to_converted_statements_map[qid] = converted_statements
    return wvs_statements_data, wvs_statements_data_map, all_QIDs, qid_to_converted_statements_map


def compile_human_labels_data(is_refined=False,
                              is_save=False):
    """
    Save human labels data.
    """
    wvs_statements_data, wvs_statements_data_map, all_QIDs, qid_to_converted_statements_map = load_statements_data(
        is_refined=is_refined)

    human_data_path = "data/WVS/WVS_Cross-National_Wave_7_csv_v5_0.csv"
    df_human_data = pd.read_csv(human_data_path)
    df_human_data = df_human_data[["D_INTERVIEW"] + all_QIDs]
    df_human_data = df_human_data.drop_duplicates(subset=["D_INTERVIEW"])
    human_data = df_human_data.to_dict(orient="records")

    human_data_selected = []
    total_valid_questions = []
    for i, row in enumerate(tqdm(human_data, desc="Compiling human labels data")):
        is_row_selected = True
        valid_questions_count = 0
        for qid in all_QIDs:
            row[f"{qid}_aid"] = row[qid]
            aid = row[qid]
            aid_candiates_flattened = wvs_statements_data_map[qid]["all_answer_ids"]
            if aid not in aid_candiates_flattened:
                is_row_selected = False
                row[qid] = -99
            else:
                valid_questions_count += 1
                answer_ids_to_grouped_answer_ids = wvs_statements_data_map[
                    qid]["answer_ids_to_grouped_answer_ids"]
                row[qid] = answer_ids_to_grouped_answer_ids[str(aid)]
        row["num_valid_questions"] = valid_questions_count
        total_valid_questions.append(valid_questions_count)

        if is_row_selected:
            human_data_selected.append(row)

    print("Total number of people: ", len(total_valid_questions))
    print("Total number of valid people: ", len(human_data_selected))
    print("Total valid questions: ", sum(total_valid_questions))
    print("Total valid questions per person (mean): ",
          "{:.2f}".format(np.mean(total_valid_questions)))
    print("Total valid questions per person (std): ",
          "{:.2f}".format(np.std(total_valid_questions)))

    if is_save:
        if is_refined:
            save_path = "data/WVS/human_labels/refined_statements_no_nan.jsonl"
            save_path_full_set = "data/WVS/human_labels/refined_statements_full_set.jsonl"
        else:
            save_path = "data/WVS/human_labels/statements_no_nan.jsonl"
            save_path_full_set = "data/WVS/human_labels/statements_full_set.jsonl"

        write_standard_data(human_data_selected, save_path)
        write_standard_data(human_data, save_path_full_set)


def load_human_labels_data(is_full_set=True, is_refined=False):
    if is_refined:
        if is_full_set:
            data_path = "data/WVS/human_labels/refined_statements_full_set.jsonl"
        else:
            data_path = "data/WVS/human_labels/refined_statements_no_nan.jsonl"
    else:
        if is_full_set:
            data_path = "data/WVS/human_labels/statements_full_set.jsonl"
        else:
            data_path = "data/WVS/human_labels/statements_no_nan.jsonl"
    return load_standard_data(data_path)


def look_at_human_labels_data(is_full_set=False, is_refined=False):
    human_labels_data = load_human_labels_data(
        is_full_set=is_full_set, is_refined=is_refined)

    for d in human_labels_data:
        pretty_print_dict(d)
        break
    return human_labels_data


def load_codebook_data():
    codebook_path = "data/WVS/codebook.json"
    with open(codebook_path, "r") as f:
        codebook = json.load(f)
    return codebook


def add_converted_demographic_data(qid,
                                   answer_vars,
                                   answer_ids_vars,
                                   template_statement,
                                   converted_statements,
                                   question_category,
                                   is_mc=False,
                                   question_id_converted=None):
    d_converted_data = {}
    d_converted_data["question_id"] = qid
    if question_id_converted is not None:
        d_converted_data["question_id_converted"] = question_id_converted
    else:
        d_converted_data["question_id_converted"] = qid

    if is_mc:
        d_converted_data["answer_ids"] = answer_ids_vars
        d_converted_data["all_answer_ids"] = []
        for aids in answer_ids_vars:
            d_converted_data["all_answer_ids"] += aids
        d_converted_data["grouped_answer_ids"] = [
            i for i in range(len(answer_ids_vars))]
        d_converted_data["grouped_answer_ids_to_answer_ids"] = {
            i: answer_ids_vars[i] for i in range(len(answer_ids_vars))}
        d_converted_data["answer_ids_to_grouped_answer_ids"] = {}
        for i, aids in enumerate(answer_ids_vars):
            for aid in aids:
                d_converted_data["answer_ids_to_grouped_answer_ids"][aid] = i
        d_converted_data["converted_statements"] = converted_statements
        d_converted_data["answer_vars"] = answer_vars
    else:
        if qid == "Q290":
            answer_vars_clean = {}
            for k, v in answer_vars.items():
                answer_vars_clean[k] = v.split(": ")[-1]
            answer_vars = answer_vars_clean
        d_converted_data["answer_vars_map"] = answer_vars
        d_converted_data["converted_statements"] = [template_statement]

    d_converted_data["statement_template"] = template_statement
    d_converted_data["question_category"] = question_category
    return d_converted_data


def compile_demographics_data(is_print=False):
    core_demographics_QIDs = wvs_conversion_data.core_demographics_QIDs
    all_demographics = wvs_conversion_data.demographics

    all_demographics_data = []
    for d in all_demographics:
        QID = d["question_id"]
        answer_ids_vars = d["answer_ids"]
        question_category = d["question_category"]
        template_statement = d["conversion_template"][0]
        answer_var_name = d["variable_names"][0]
        template_statement_clean = template_statement.replace(
            f"{answer_var_name}", "answer_var")
        # is_core = (QID in core_demographics_QIDs)

        if type(answer_ids_vars) == list:
            answer_vars = d[answer_var_name]

            converted_statements = []
            for a in answer_vars:
                filled_template = template_statement.replace(
                    "{" + answer_var_name + "}", a)
                converted_statements.append(filled_template)

            d_converted_data = add_converted_demographic_data(QID,
                                                              answer_vars,
                                                              answer_ids_vars,
                                                              template_statement_clean,
                                                              converted_statements,
                                                              question_category,
                                                              is_mc=True)

            all_demographics_data.append(d_converted_data)
            if is_print:
                pretty_print_dict(d_converted_data)

        elif QID == "B_COUNTRY_to_continent":
            country_name_to_continent_name = wvs_conversion_data.country_name_to_continent_name
            for demographics_obj in all_demographics:
                if demographics_obj["question_id"] == "B_COUNTRY":
                    country_obj = demographics_obj
                    break

            country_id_to_country_name = country_obj["original_answer_options"]
            country_id_to_continent_name = {}
            for country_id, country_name in country_id_to_country_name.items():
                continent_name = country_name_to_continent_name[country_name]
                country_id_to_continent_name[country_id] = continent_name

            d_converted_data = add_converted_demographic_data(QID,
                                                              country_id_to_continent_name,
                                                              "",
                                                              template_statement_clean,
                                                              "",
                                                              question_category,
                                                              is_mc=False,
                                                              question_id_converted="B_COUNTRY")
            all_demographics_data.append(d_converted_data)
            if is_print:
                pretty_print_dict(d_converted_data)
        else:
            answer_vars = d[answer_var_name]
            d_converted_data = add_converted_demographic_data(QID,
                                                              answer_vars,
                                                              "",
                                                              template_statement_clean,
                                                              "",
                                                              question_category,
                                                              is_mc=False)
            all_demographics_data.append(d_converted_data)
            if is_print:
                pretty_print_dict(d_converted_data)

    save_path = "data/WVS/meta_data/demographics_meta_data.jsonl"
    write_standard_data(all_demographics_data, save_path)


def compile_human_labels_data_with_demographics():
    all_demographics_QIDs = wvs_conversion_data.all_demographics_QIDs

    save_path = "data/WVS/meta_data/demographics_meta_data.jsonl"
    converted_demographics_data = load_standard_data(save_path)
    converted_demographics_data_map = {
        d["question_id"]: d for d in converted_demographics_data}

    human_data_path = "data/WVS/WVS_Cross-National_Wave_7_csv_v5_0.csv"
    df_human_data = pd.read_csv(human_data_path)
    df_human_data = df_human_data.drop_duplicates(subset=["D_INTERVIEW"])

    selected_cols = ["D_INTERVIEW"] + all_demographics_QIDs
    selected_cols_intersection = list(set(df_human_data.columns).intersection(
        selected_cols))
    df_human_data = df_human_data[selected_cols_intersection]
    human_data = df_human_data.to_dict(orient="records")

    all_human_data_with_demographics_statements = []
    for d in tqdm(human_data):
        human_data_with_demographics_statements = {
            "D_INTERVIEW": d["D_INTERVIEW"]}

        num_valid_questions = 0
        for qid in all_demographics_QIDs:
            qid_converted = converted_demographics_data_map[qid]["question_id_converted"]
            a_id = d[qid_converted]
            grouped_aid = -99
            statement = -99

            if "answer_vars_map" in converted_demographics_data_map[qid]:
                q_answer_vars_map = converted_demographics_data_map[qid]["answer_vars_map"]
                if str(a_id) in q_answer_vars_map:
                    grouped_aid = a_id
                    statement_template = converted_demographics_data_map[qid]["statement_template"]
                    statement = statement_template.replace(
                        "{answer_var}", q_answer_vars_map[str(a_id)])
                    num_valid_questions += 1
            else:
                if a_id in converted_demographics_data_map[qid]["all_answer_ids"]:
                    grouped_aid = converted_demographics_data_map[qid]["answer_ids_to_grouped_answer_ids"][str(
                        a_id)]
                    statement = converted_demographics_data_map[qid]["converted_statements"][grouped_aid]
                    num_valid_questions += 1

            d[qid] = grouped_aid
            d[f"{qid}_aid"] = a_id
            human_data_with_demographics_statements[qid] = statement
            human_data_with_demographics_statements[f"{qid}_aid"] = a_id

        d["num_valid_demographics"] = num_valid_questions
        human_data_with_demographics_statements["num_valid_demographics"] = num_valid_questions

        all_human_data_with_demographics_statements.append(
            human_data_with_demographics_statements)

    write_standard_data(
        human_data, "data/WVS/human_labels/demographics.jsonl")
    write_standard_data(all_human_data_with_demographics_statements,
                        "data/WVS/human_labels/demographics_in_nl.jsonl")


def merge_statements_and_human_labels_data(is_refined=True):
    if is_refined:
        statements_data = load_standard_data("data/WVS/human_labels/refined_statements_full_set.jsonl")
        statements_data_no_nan = load_standard_data("data/WVS/human_labels/refined_statements_no_nan.jsonl")
    else:
        statements_data = load_standard_data("data/WVS/human_labels/statements_full_set.jsonl")
        statements_data_no_nan = load_standard_data("data/WVS/human_labels/statements_no_nan.jsonl")

    demographics_data = load_standard_data("data/WVS/human_labels/demographics.jsonl")
    demographics_in_nl_data = load_standard_data("data/WVS/human_labels/demographics_in_nl.jsonl")

    df_statements_data = pd.DataFrame(statements_data)
    df_statements_data_no_nan = pd.DataFrame(statements_data_no_nan)
    df_demographics_data = pd.DataFrame(demographics_data)
    df_demographics_in_nl_data = pd.DataFrame(demographics_in_nl_data)

    refined_string = "refined_" if is_refined else ""
    df_joined = df_statements_data.merge(df_demographics_data, on="D_INTERVIEW")
    save_path = f"data/WVS/human_labels/demographics_{refined_string}statements_combined_full_set.jsonl"
    write_standard_data(df_joined.to_dict(orient="records"), save_path)

    df_joined = df_statements_data.merge(df_demographics_in_nl_data, on="D_INTERVIEW")
    save_path = f"data/WVS/human_labels/demographics_in_nl_{refined_string}statements_combined_full_set.jsonl"
    write_standard_data(df_joined.to_dict(orient="records"), save_path)

    df_joined = df_statements_data_no_nan.merge(df_demographics_data, on="D_INTERVIEW")
    df_joined = df_joined[df_joined["D_INTERVIEW"].isin(df_statements_data_no_nan["D_INTERVIEW"])]
    save_path = f"data/WVS/human_labels/demographics_{refined_string}statements_combined_no_nan.jsonl"
    write_standard_data(df_joined.to_dict(orient="records"), save_path)

    df_joined = df_statements_data_no_nan.merge(df_demographics_in_nl_data, on="D_INTERVIEW")
    df_joined = df_joined[df_joined["D_INTERVIEW"].isin(df_statements_data_no_nan["D_INTERVIEW"])]
    save_path = f"data/WVS/human_labels/demographics_in_nl_{refined_string}statements_combined_no_nan.jsonl"
    write_standard_data(df_joined.to_dict(orient="records"), save_path)


if __name__ == "__main__":
    # compile_statements_data(is_refined=True)
    # compile_statements_data(is_refined=False)
    # compile_human_labels_data(is_refined=True, is_save=True)
    # compile_human_labels_data(is_refined=False, is_save=True)

    # compile_demographics_data(is_print=True)
    # compile_human_labels_data_with_demographics()
    merge_statements_and_human_labels_data()
