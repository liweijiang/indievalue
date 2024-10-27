import numpy as np
from tqdm import tqdm
from src.utils.main_utils import *
from src.utils.chat_models import *

from data.WVS.WVS_conversion import *
from probe_utils import *


def get_random_baseline_results(model_name="gpt-4o-2024-08-06",
                                num_stmts=200,
                                probe_setup_id=0,
                                mode="stmt",
                                data_version="090824_800_refined"):
    expt_id = f"{mode}-{data_version}-v{probe_setup_id}-num_stmt_{num_stmts}"
    data_path = f"data/WVS/probe_expts/{data_version}/{expt_id}.jsonl"
    results_path = data_path.replace(
        "data/WVS/", "results/").replace(f"{expt_id}.jsonl", f"{model_name}/{expt_id}.jsonl")
    results_data = load_standard_data(results_path, is_print=False)

    probe_setup_QIDs = get_all_probe_qids(
        probe_setup_id=probe_setup_id)
    statements_metadata_map = get_statements_metadata_map(is_refined=False)

    all_accs = []
    for d in tqdm(results_data):
        for qid in probe_setup_QIDs:
            for i in range(5000):
                agid = d[qid]
                random_aid = random.choice(
                    [i for i in range(len(statements_metadata_map[qid]["converted_statements"]))])

                all_accs.append(int(agid == random_aid))

    print("Mean Acc: ", np.mean(all_accs), np.std(all_accs))


def main(model_name="gpt-4o-2024-08-06",
         num_stmts=200,
         probe_setup_id=0,
         mode="stmt",
         data_version="090824_800",
         is_save=True):
    expt_id = f"{mode}-{data_version}-v{probe_setup_id}-num_stmt_{num_stmts}"
    data_path = f"data/WVS/probe_expts/{data_version}/{expt_id}.jsonl"
    results_path = data_path.replace(
        "data/WVS/", "results/").replace(f"{expt_id}.jsonl", f"{model_name}/{expt_id}.jsonl")
    results_data = load_standard_data(results_path, is_print=False)
    print(results_path)

    probe_setup_QIDs = get_all_probe_qids(
        probe_setup_id=probe_setup_id)

    all_accs = []
    all_data_to_save = []
    for d in results_data:
        if "raw_response" not in d:
            continue
        parsed_d_results = parse_response(d["raw_response"])
        if parsed_d_results == "":
            continue

        if len(parsed_d_results) != len(probe_setup_QIDs):
            raise ValueError(f"Error:", len(parsed_d_results),
                             len(probe_setup_QIDs))

        d_accs = []
        for probe_qid, probe_formatted_id in zip(probe_setup_QIDs, parsed_d_results):
            probe_result = parsed_d_results[probe_formatted_id]
            ref_gaid = d[probe_qid]
            probe_gaid = probe_result["choice_grouped_answer_id"]
            d_accs.append(ref_gaid == probe_gaid)
            d[f"{probe_qid}_is_correct"] = int(ref_gaid == probe_gaid)
        all_data_to_save.append(d)
        all_accs.append(np.mean(d_accs))

    print(len(all_accs))
    print("Mean Acc: ", np.mean(all_accs), np.std(all_accs))

    if is_save:
        save_path = results_path.replace(
            f"{expt_id}.jsonl", f"is_correct/{expt_id}.jsonl")
        write_standard_data(all_data_to_save, save_path)


def main_refined(model_name="gpt-4o-2024-08-06",
                 num_stmts=200,
                 probe_setup_id=0,
                 mode="stmt",
                 data_version="probe_data_090524_800",
                 is_save=True):
    expt_id = f"{mode}-{data_version}-v{probe_setup_id}-num_stmt_{num_stmts}"
    data_path = f"data/WVS/probe_expts/{data_version}/{expt_id}.jsonl"
    results_path = data_path.replace(
        "data/WVS/", "results/").replace(f"{expt_id}.jsonl", f"{model_name}/{expt_id}.jsonl")
    results_data = load_standard_data(results_path, is_print=False)

    statements_meta_data = get_statements_metadata_map(is_refined=True)
    statements_meta_data_polar = get_statements_metadata_map(is_refined=False)

    probe_setup_QIDs = get_all_probe_qids(probe_setup_id=probe_setup_id)

    all_accs = []
    all_data_to_save = []
    for d in results_data:
        if "raw_response" not in d:
            continue
        parsed_d_results = parse_response(d["raw_response"])
        if parsed_d_results == "":
            continue

        if len(parsed_d_results) != len(probe_setup_QIDs):
            raise ValueError(f"Error:", len(parsed_d_results),
                             len(probe_setup_QIDs))

        d_accs = []
        for probe_qid, probe_formatted_id in zip(probe_setup_QIDs, parsed_d_results):
            probe_result = parsed_d_results[probe_formatted_id]
            ref_aid = d[f"{probe_qid}_aid"]
            ref_gaid = statements_meta_data_polar[probe_qid]["answer_ids_to_grouped_answer_ids"][str(
                ref_aid)]
            probe_gaid = probe_result["choice_grouped_answer_id"]
            d_accs.append(ref_gaid == probe_gaid)
            d[f"{probe_qid}_is_correct"] = int(ref_gaid == probe_gaid)
        all_data_to_save.append(d)
        all_accs.append(np.mean(d_accs))

    print(len(all_accs))
    print("Mean Acc: ", np.mean(all_accs), np.std(all_accs))

    if is_save:
        save_path = results_path.replace(
            f"{expt_id}.jsonl", f"is_correct/{expt_id}.jsonl")
        write_standard_data(all_data_to_save, save_path)


def get_evenness_index_of_models(model_name="gpt-4o-2024-08-06",
                                 num_stmts=200,
                                 mode="stmt",  # stmt, demographics, stmt_demographics
                                 data_version="090824_800",
                                 demographics_qid_to_plot=core_demographics_QIDs_to_demographics_name.keys()):
    all_acc = {}
    for probe_setup_id in range(3):
        expt_id = f"{mode}-{data_version}-v{probe_setup_id}-num_stmt_{num_stmts}"
        data_path = f"data/WVS/probe_expts/{data_version}/{expt_id}.jsonl"
        print(f"{model_name}/{expt_id}.jsonl")
        results_path = data_path.replace(
            "data/WVS/", "results/").replace(f"{expt_id}.jsonl", f"{model_name}/is_correct/{expt_id}.jsonl")
        results_data = load_standard_data(results_path, is_print=False)
        probe_qids = get_all_probe_qids(probe_setup_id=probe_setup_id)
        demographics_metadata = get_demographics_metadata_map()

        for demographics_qid in demographics_qid_to_plot:
            acc_by_demographics_dimension = {}
            for d in results_data:
                demographics_dimension = d[demographics_qid]
                if demographics_dimension == -99:
                    continue
                if demographics_dimension not in acc_by_demographics_dimension:
                    acc_by_demographics_dimension[demographics_dimension] = []
                d_acc = get_accuracy(d, probe_qids)
                acc_by_demographics_dimension[demographics_dimension].append(
                    d_acc)

            if demographics_qid not in all_acc:
                all_acc[demographics_qid] = acc_by_demographics_dimension
            else:
                for demographics_dimension in acc_by_demographics_dimension:
                    if demographics_dimension not in all_acc[demographics_qid]:
                        all_acc[demographics_qid][demographics_dimension] = []
                    else:                                                                           
                        all_acc[demographics_qid][demographics_dimension].extend(
                            acc_by_demographics_dimension[demographics_dimension])

    for demographics_qid in all_acc:
        print("=" * 30, demographics_qid, "=" * 30)
        print(list(all_acc[demographics_qid].keys()))

    for i, (demographics_qid, acc_by_demographics_dimension) in enumerate(all_acc.items()):
        converted_statements = demographics_metadata[demographics_qid]["converted_statements"]

        plot_data = []
        for dimension, accuracies in acc_by_demographics_dimension.items():
            plot_data.extend([(dimension, acc * 100) for acc in accuracies])

        # Create a DataFrame
        df = pd.DataFrame(plot_data, columns=['Demographic Dimension', 'Accuracy'])


        df_by_cat = df.groupby(['Demographic Dimension'])['Accuracy'].mean()
        stds = df_by_cat.std()

        print(stds)
        # print("Mean:", df_by_cat)


if __name__ == "__main__":
    # add argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="gpt-4o-2024-08-06")
    parser.add_argument("--num_stmts", type=int, default=200)
    parser.add_argument("--probe_setup_id", type=int, default=0)
    parser.add_argument("--mode", type=str, default="stmt")
    parser.add_argument("--data_version", type=str,
                        default="090824_800")
    parser.add_argument("--is_save", type=bool, default=True)
    args = parser.parse_args()

    main(model_name=args.model_name,
         num_stmts=args.num_stmts,
         probe_setup_id=args.probe_setup_id,
         mode=args.mode,
         data_version=args.data_version,
         is_save=args.is_save)

    # get_evenness_index_of_models(model_name=args.model_name,
    #                              num_stmts=args.num_stmts,
    #                              mode=args.mode,
    #                              data_version=args.data_version)

    # for probe_setup_id in [1]: # , 1, 2
    #     for num_stmts in [200]: # 0, 50, 100, 150,
    #         print("=" * 50)
    #         print(f"Probe Setup ID: {probe_setup_id}, Num Stmts: {num_stmts}")
    #         main(model_name=args.model_name,
    #              num_stmts=num_stmts,
    #              probe_setup_id=probe_setup_id,
    #              mode=args.mode,
    #              data_version=args.data_version)

    # if "refined" in args.data_version:
    #     main_refined(model_name=args.model_name,
    #                  num_stmts=args.num_stmts,
    #                  probe_setup_id=args.probe_setup_id,
    #                  mode=args.mode,
    #                  data_version=args.data_version,
    #                  is_save=True)
    # else:
    #     main(model_name=args.model_name,
    #          num_stmts=args.num_stmts,
    #          probe_setup_id=args.probe_setup_id,
    #          mode=args.mode,
    #          data_version=args.data_version,
    #          is_save=True)

    # get_random_baseline_results(probe_setup_id=0)
    # get_random_baseline_results(probe_setup_id=1)
    # get_random_baseline_results(probe_setup_id=2)

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
    #     for probe_setup_id in range(3):
    #         main("gpt-4o-2024-08-06",
    #              num_stmts=num_stmts,
    #              probe_setup_id=probe_setup_id,
    #              mode="stmt",
    #              data_version="090824_800_sample_100",
    #              is_save=True)


# "claude-3-5-sonnet-20240620"
# "mistralai/Mixtral-8x22B-Instruct-v0.1"
# "mistralai/Mixtral-8x7B-Instruct-v0.1"
# "gpt-4o-2024-08-06",
# "gpt-4o-2024-05-13",
# "gpt-4o-mini-2024-07-18",
# "gpt-4-turbo-2024-04-09",
# "gpt-3.5-turbo-0125",
# "meta-llama/Meta-Llama-3.1-8B-Instruct"
# "meta-llama/Meta-Llama-3.1-70B-Instruct"
