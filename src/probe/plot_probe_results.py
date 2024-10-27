import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from probe_utils import *
from scipy import stats
from data.WVS.WVS_conversion import *


def main_demographics_number_of_statements(model_name="gpt-4o-2024-08-06", data_version="090824_800", num_probe_setups=3):
    num_stmts_accs = {}
    for num_stmts in [0, 50, 100, 150, 200]:
        mode_accs = {}
        for mode in ["stmt", "stmt_demographics"]:
            all_accs = []
            for probe_setup_id in range(num_probe_setups):
                probe_setup_QIDs = get_all_probe_qids(
                    probe_setup_id=probe_setup_id)
                expt_id = f"{mode}-{data_version}-v{probe_setup_id}-num_stmt_{num_stmts}"
                data_path = f"data/WVS/probe_expts/{data_version}/{expt_id}.jsonl"
                print(f"{model_name}/{expt_id}.jsonl")
                results_path = data_path.replace(
                    "data/WVS/", "results/").replace(f"{expt_id}.jsonl", f"{model_name}/{expt_id}.jsonl")
                results_data = load_standard_data(results_path, is_print=False)

                for j, d in enumerate(results_data):
                    parsed_d_results = parse_response(d["raw_response"])

                    ref_aids = []
                    probe_aids = []
                    for probe_qid, probe_formatted_id in zip(probe_setup_QIDs, parsed_d_results):
                        probe_result = parsed_d_results[probe_formatted_id]

                        ref_aid = d[probe_qid]
                        probe_aid = probe_result["choice_grouped_answer_id"]

                        ref_aids.append(ref_aid)
                        probe_aids.append(probe_aid)

                    acc = np.mean(np.array(ref_aids) ==
                                  np.array(probe_aids)) * 100
                    if j >= len(all_accs):
                        all_accs.append(acc)
                    else:
                        all_accs[j] += acc

            mode_accs[mode] = [acc / num_probe_setups for acc in all_accs]

        num_stmts_accs[num_stmts] = mode_accs

    # test statistic significance
    for num_stmts in num_stmts_accs:
        stmt_accs = num_stmts_accs[num_stmts]['stmt']
        stmt_demographics_accs = num_stmts_accs[num_stmts]['stmt_demographics']
        t_statistic, p_value = stats.ttest_ind(
            stmt_accs, stmt_demographics_accs)
        print(
            f"num_stmts: {num_stmts}, P-value between Statements and Demographics accuracies: {p_value:.12f}")

        only_demographics_accs = num_stmts_accs[0]['stmt_demographics']

        t_statistic, p_value = stats.ttest_ind(
            stmt_accs, only_demographics_accs)
        print(
            f"num_stmts: {num_stmts}, P-value between Statements and Only Demographics accuracies: {p_value:.12f}")

    # Create a DataFrame from the num_stmts_accs dictionary
    df = pd.DataFrame([
        {'Num_Stmts': num_stmts, 'Mode': mode, 'Accuracy': acc}
        for num_stmts, mode_accs in num_stmts_accs.items()
        for mode in ['stmt', 'stmt_demographics']
        for acc in mode_accs[mode]
    ])

    # Set up the plot style
    plt.figure(figsize=(5.5, 4))
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # Increase font size
    plt.rcParams.update({'font.size': 12})

    # Create the box plot with closer groups
    ax = sns.boxplot(x='Num_Stmts',
                     y='Accuracy',
                     hue='Mode',
                     data=df,
                     width=0.8,  # Keep the individual bar width the same
                     fliersize=4,
                     dodge=0.7)  # Reduce the dodge parameter to bring groups closer

    # Customize the plot
    plt.xlabel("Number of Value Statements")
    plt.ylabel("Accuracy")
    # plt.title(f"Probe Accuracy Distribution for {model_name}\n(probe_setup_id={probe_setup_id})")

    # Adjust colors for better visibility
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'

    # Add average labels
    for i, num_stmts in enumerate(df['Num_Stmts'].unique()):
        for j, mode in enumerate(['stmt', 'stmt_demographics']):
            avg = df[(df['Num_Stmts'] == num_stmts) & (
                df['Mode'] == mode)]['Accuracy'].mean()
            median = df[(df['Num_Stmts'] == num_stmts) & (
                df['Mode'] == mode)]['Accuracy'].median()
            # Position the label above both the average and median
            y_pos = max(avg, median) + 0.5
            # Increased fontsize
            ax.text(i + (j-0.5)*0.4, y_pos,
                    f'{avg:.2f}', ha='center', va='bottom', fontsize=8)

    # Add red dotted horizontal line at the average of the first bar
    first_bar_avg = df[(df['Num_Stmts'] == 0) & (
        df['Mode'] == 'stmt_demographics')]['Accuracy'].mean()
    ax.axhline(y=first_bar_avg, color='red',
               linestyle='--', linewidth=1.2, zorder=11)

    # Add a line for random baseline
    ax.axhline(y=45.37, color='blue', linestyle='--', linewidth=1.2)

    # Add mean lines to each box
    for i, num_stmts in enumerate(df['Num_Stmts'].unique()):
        for j, mode in enumerate(['stmt', 'stmt_demographics']):
            mean_val = df[(df['Num_Stmts'] == num_stmts) & (
                df['Mode'] == mode)]['Accuracy'].mean()
            ax.hlines(y=mean_val, xmin=i+j*0.4-0.12 - 0.2, xmax=i +
                      j*0.4+0.12 - 0.2, color='gold', linewidth=1, zorder=10)

    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.legend().remove()
    plt.tight_layout()

    # set y limit
    ax.set_ylim(28.5, 92)

    # Save the plot in high resolution
    save_dir = f"results/plots"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        f"{save_dir}/box_plot_by_num_stmts_probe_setup_{data_version}_{model_name}.png",
        bbox_inches='tight',
        dpi=300,
        format='png'
    )
    plt.close()

    print(
        f"High-resolution box plot saved to {save_dir}/box_plot_by_num_stmts_probe_setup_{probe_setup_id}.png")


def load_results_data_is_correct(model_name,
                                 num_stmts,
                                 probe_setup_id,
                                 mode,
                                 data_version):
    expt_id = f"{mode}-{data_version}-v{probe_setup_id}-num_stmt_{num_stmts}"
    data_path = f"data/WVS/probe_expts/{data_version}/{expt_id}.jsonl"
    print(f"{model_name}/{expt_id}.jsonl")
    results_path = data_path.replace(
        "data/WVS/", "results/").replace(f"{expt_id}.jsonl", f"{model_name}/is_correct/{expt_id}.jsonl")
    results_data = load_standard_data(results_path, is_print=False)
    return results_data


def get_accuracy_by_probe_cat(model_name,
                              num_stmts,
                              mode,
                              data_version,
                              all_accs_by_probe_cat):
    accs_by_probe_cat = {}
    for probe_setup_id in range(3):
        probe_setup_QIDs_by_cat = get_probe_setups()[probe_setup_id]
        results_data = load_results_data_is_correct(model_name,
                                                    num_stmts,
                                                    probe_setup_id,
                                                    mode,
                                                    data_version)

        for probe_cat in probe_setup_QIDs_by_cat:
            probe_setup_QIDs = probe_setup_QIDs_by_cat[probe_cat]
            if probe_cat not in accs_by_probe_cat:
                accs_by_probe_cat[probe_cat] = []
            for d in results_data:
                d_acc = get_accuracy(d, probe_setup_QIDs)
                accs_by_probe_cat[probe_cat].append(d_acc)

    overall_accs = []
    for probe_cat in accs_by_probe_cat:
        overall_accs.extend(accs_by_probe_cat[probe_cat])
        accs_by_probe_cat[probe_cat] = np.mean(
            accs_by_probe_cat[probe_cat]) * 100
    accs_by_probe_cat["Overall"] = np.mean(overall_accs) * 100
    all_accs_by_probe_cat.append(accs_by_probe_cat)
    return all_accs_by_probe_cat


def get_accuracy_by_probe_cat_with_demo_count(model_name,
                                              num_stmts,
                                              mode,
                                              data_version,
                                              all_accs_by_probe_cat,
                                              demo_probe_cat):
    accs_by_probe_cat = {}
    for probe_setup_id in range(3):
        probe_setup_QIDs_by_cat = get_probe_setups()[probe_setup_id]
        results_data = load_results_data_is_correct(model_name,
                                                    num_stmts,
                                                    probe_setup_id,
                                                    mode,
                                                    data_version)

        for probe_cat in probe_setup_QIDs_by_cat:
            probe_setup_QIDs = probe_setup_QIDs_by_cat[probe_cat]
            if probe_cat not in accs_by_probe_cat:
                accs_by_probe_cat[probe_cat] = []
            for d in results_data:
                d_acc = get_accuracy(d, probe_setup_QIDs)
                accs_by_probe_cat[probe_cat].append(d_acc)

    overall_accs = []
    for probe_cat in accs_by_probe_cat:
        overall_accs.extend(accs_by_probe_cat[probe_cat])
        accs_by_probe_cat[probe_cat] = np.mean(
            accs_by_probe_cat[probe_cat]) * 100
   
    # statements_qids_by_category = get_statements_qids_by_category()
    # accs_by_probe_cat["Count"] = len(statements_qids_by_category[demo_probe_cat]) - 3

    all_accs_by_probe_cat.append(accs_by_probe_cat)
    return all_accs_by_probe_cat


def get_random_baseline_accuracy_by_probe_cat(data_version, all_accs_by_probe_cat):
    accs_by_probe_cat = {}
    for probe_setup_id in range(3):
        results_data = load_results_data_is_correct("gpt-4o-2024-08-06",
                                                    200,
                                                    probe_setup_id,
                                                    "stmt",
                                                    data_version)

        probe_setup_QIDs_by_cat = get_probe_setups()[probe_setup_id]
        statements_metadata_map = get_statements_metadata_map(is_refined=False)

        for d in tqdm(results_data):
            for probe_cat in probe_setup_QIDs_by_cat:
                if probe_cat not in accs_by_probe_cat:
                    accs_by_probe_cat[probe_cat] = []
                probe_setup_QIDs = probe_setup_QIDs_by_cat[probe_cat]
                for qid in probe_setup_QIDs:
                    for i in range(500):
                        agid = d[qid]
                        random_aid = random.choice(
                            [i for i in range(len(statements_metadata_map[qid]["converted_statements"]))])
                        accs_by_probe_cat[probe_cat].append(
                            int(agid == random_aid))

    overall_accs = []
    for probe_cat in accs_by_probe_cat:
        overall_accs.extend(accs_by_probe_cat[probe_cat])
        accs_by_probe_cat[probe_cat] = np.mean(
            accs_by_probe_cat[probe_cat]) * 100

    accs_by_probe_cat["Overall"] = np.mean(overall_accs) * 100

    all_accs_by_probe_cat.append(accs_by_probe_cat)
    return all_accs_by_probe_cat


def main_compare_models():
    ############ configs ############
    list_of_model_names = [
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4-turbo-2024-04-09",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "Qwen/Qwen2-72B-Instruct",
        "claude-3-5-sonnet-20240620"
    ]

    model_name_map = {
        "gpt-4-turbo-2024-04-09": "GPT-4-turbo (0409)",
        "gpt-4o-2024-05-13": "GPT-4o (0513)",
        "gpt-4o-2024-08-06": "GPT-4o (0806)",
        "gpt-4o-mini-2024-07-18": "GPT-4o-mini (0718)",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "LLama-3.1-8B",
        "meta-llama/Meta-Llama-3.1-70B-Instruct": "LLama-3.1-70B",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
        "mistralai/Mixtral-8x22B-Instruct-v0.1": "Mixtral-8x22B",
        "Qwen/Qwen2-72B-Instruct": "Qwen2-72B",
        "claude-3-5-sonnet-20240620": "Claude-3.5 (Sonnet)",
        "GPT-4o (0806) Rand": "GPT-4o (0806) Rand",
        "Random": "Random"
    }
    probe_cat_name_map = {
        "Social Values, Attitudes & Stereotypes": "Social Values & Stereotypes",
        "Happiness and Well-Being": "Happiness & Well-Being",
        "Social Capital, Trust & Organizational Membership": "Social Capital & Trust",
        "Economic Values": "Economic Values",
        "Corruption": "Corruption",
        "Migration": "Migration",
        "Security": "Security",
        "Postmaterialist Index": "Postmaterialist Index",
        "Science & Technology": "Science & Technology",
        "Religious Values": "Religious Values",
        "Ethical Values and Norms": "Ethical Values & Norms",
        "Political Interest & Political Participation": "Political Interest & Participation",
        "Political Culture & Political Regimes": "Political Culture & Regimes"
    }

    num_stmts = 200
    mode = "stmt"
    data_version = "090824_800"
    ############ compute accuracy by probe cat ############
    all_accs_by_probe_cat = []

    all_accs_by_probe_cat = get_random_baseline_accuracy_by_probe_cat(
        data_version, all_accs_by_probe_cat)

    all_accs_by_probe_cat = get_accuracy_by_probe_cat("gpt-4o-2024-08-06",
                                                      0,
                                                      "stmt",
                                                      data_version,
                                                      all_accs_by_probe_cat)

    for model_name in list_of_model_names:
        all_accs_by_probe_cat = get_accuracy_by_probe_cat(model_name,
                                                          num_stmts,
                                                          mode,
                                                          data_version,
                                                          all_accs_by_probe_cat)

    list_of_model_names.insert(0, "GPT-4o (0806) Rand")
    list_of_model_names.insert(0, "Random")

    df_all_accs_by_probe_cat = pd.DataFrame(all_accs_by_probe_cat)
    df_all_accs_by_probe_cat = df_all_accs_by_probe_cat.rename(
        columns=probe_cat_name_map)

    ############ plot ############
    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(df_all_accs_by_probe_cat.T, annot=True, cmap="rocket_r",
                     fmt=".1f", cbar_kws={}, annot_kws={"size": 12}, vmin=30, vmax=100)
    plt.tick_params(axis='both', which='major', labelsize=12)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    plt.xticks(np.arange(len(list_of_model_names)) + 0.5,
               [model_name_map[model_name] for model_name in list_of_model_names], rotation=90, ha='center')
    plt.tight_layout()
    plt.savefig("results/plots/probe/heatmap/probe_results_heatmap.png", dpi=1000, bbox_inches='tight')
    plt.close()

    print("Heatmap saved as 'probe_results_heatmap.png'")


def plot_heatmap_broken_down_by_category():
    data_to_gen = ["social_values,_attitudes_&_stereotypes",
                   "happiness_and_well-being",
                   "social_capital,_trust_&_organizational_membership",
                   "economic_values",
                   "corruption",
                   "migration",
                   "security",
                   "postmaterialist_index",
                   "science_&_technology",
                   "religious_values",
                   "ethical_values_and_norms",
                   "political_interest_&_political_participation",
                   "political_culture_&_political_regimes"]

    probe_cat_name_map = {
        "social_values,_attitudes_&_stereotypes": "Social Values & Stereotypes",
        "happiness_and_well-being": "Happiness & Well-Being",
        "social_capital,_trust_&_organizational_membership": "Social Capital & Trust",
        "economic_values": "Economic Values",
        "corruption": "Corruption",
        "migration": "Migration",
        "security": "Security",
        "postmaterialist_index": "Postmaterialist Index",
        "science_&_technology": "Science & Technology",
        "religious_values": "Religious Values",
        "ethical_values_and_norms": "Ethical Values & Norms",
        "political_interest_&_political_participation": "Political Interest & Participation",
        "political_culture_&_political_regimes": "Political Culture & Regimes"
    }

    probe_cat_name_full_map = {
        "social_values,_attitudes_&_stereotypes": "Social Values, Attitudes & Stereotypes",
        "happiness_and_well-being": "Happiness and Well-Being",
        "social_capital,_trust_&_organizational_membership": "Social Capital, Trust & Organizational Membership",
        "economic_values": "Economic Values",
        "corruption": "Corruption",
        "migration": "Migration",
        "security": "Security",
        "postmaterialist_index": "Postmaterialist Index",
        "science_&_technology": "Science & Technology",
        "religious_values": "Religious Values",
        "ethical_values_and_norms": "Ethical Values and Norms",
        "political_interest_&_political_participation": "Political Interest & Political Participation",
        "political_culture_&_political_regimes": "Political Culture & Political Regimes"
    }

    all_accs_by_probe_cat = []
    for num_stmts in data_to_gen:
        all_accs_by_probe_cat = get_accuracy_by_probe_cat_with_demo_count(model_name="gpt-4o-2024-08-06",
                                                                          num_stmts=num_stmts,
                                                                          mode="stmt",
                                                                          data_version="090824_800_sample_100",
                                                                          all_accs_by_probe_cat=all_accs_by_probe_cat,
                                                                          demo_probe_cat=probe_cat_name_full_map[num_stmts])

    df_all_accs_by_probe_cat = pd.DataFrame(all_accs_by_probe_cat)
    # Move 'Count' to the first column
    # cols = ['Count'] + [col for col in df_all_accs_by_probe_cat.columns if col != 'Count']
    # df_all_accs_by_probe_cat = df_all_accs_by_probe_cat[cols]
    df_all_accs_by_probe_cat = df_all_accs_by_probe_cat.rename(
        columns=probe_cat_name_map)

    count_list = [42, 8, 41, 3, 6, 7, 18, 3, 3, 9, 20, 32, 22]

    ############ plot ############
    plt.figure(figsize=(11, 8.5))
    ax = sns.heatmap(df_all_accs_by_probe_cat.T, annot=True, cmap="crest",
                     fmt=".1f", annot_kws={"size": 12}, vmin=30, vmax=100, cbar=False)
    plt.tick_params(axis='both', which='major', labelsize=13)

    plt.xticks(np.arange(len(data_to_gen)) + 0.5,
               [probe_cat_name_map[data_type] + f" (N={c})" for data_type, c in zip(data_to_gen, count_list)], rotation=90, ha='center')
    plt.yticks(np.arange(len(probe_cat_name_map)) + 0.5, list(probe_cat_name_map.values()), rotation=0)
    plt.tight_layout()
    plt.savefig("results/plots/probe/heatmap/probe_results_heatmap_by_category.png",
                dpi=1000, bbox_inches='tight')
    plt.close()

    print("Heatmap saved as 'results/plots/heatmap/probe_results_heatmap_by_category.png'")


def plot_by_demographics_dimensions(model_name="gpt-4o-2024-08-06",
                                    num_stmts=200,
                                    mode="stmt",  # stmt, demographics, stmt_demographics
                                    data_version="090824_800",
                                    demographics_qid_to_plot=core_demographics_QIDs_to_demographics_name.keys()):

    convert_stmt_to_short_form = {
        "B_COUNTRY": {
            'I am currently in Libya': 'Libya',
            'I am currently in Canada': 'Canada',
            'I am currently in Slovakia': 'Slovakia',
            'I am currently in Malaysia': 'Malaysia',
            'I am currently in Zimbabwe': 'Zimbabwe',
            'I am currently in Ethiopia': 'Ethiopia',
            'I am currently in Morocco': 'Morocco',
            'I am currently in Kenya': 'Kenya',
            'I am currently in Taiwan ROC': 'Taiwan ROC',
            'I am currently in Chile': 'Chile',
            'I am currently in South Korea': 'South Korea',
            'I am currently in Puerto Rico': 'Puerto Rico',
            'I am currently in Russia': 'Russia',
            'I am currently in United States': 'United States',
            'I am currently in Argentina': 'Argentina',
            'I am currently in Indonesia': 'Indonesia',
            'I am currently in Tunisia': 'Tunisia',
            'I am currently in Serbia': 'Serbia',
            'I am currently in New Zealand': 'New Zealand',
            'I am currently in Colombia': 'Colombia',
            'I am currently in Brazil': 'Brazil',
            'I am currently in Philippines': 'Philippines',
            'I am currently in Romania': 'Romania',
            'I am currently in Mexico': 'Mexico',
            'I am currently in Peru': 'Peru',
            'I am currently in Bolivia': 'Bolivia',
            'I am currently in Mongolia': 'Mongolia',
            'I am currently in Ecuador': 'Ecuador',
            'I am currently in Nicaragua': 'Nicaragua',
            'I am currently in Hong Kong SAR': 'Hong Kong SAR',
            'I am currently in Greece': 'Greece',
            'I am currently in Cyprus': 'Cyprus',
            'I am currently in Nigeria': 'Nigeria',
            'I am currently in Bangladesh': 'Bangladesh',
            'I am currently in Guatemala': 'Guatemala',
            'I am currently in Thailand': 'Thailand',
            'I am currently in Armenia': 'Armenia',
            'I am currently in Macao SAR': 'Macao SAR',
            'I am currently in Netherlands': 'Netherlands',
            'I am currently in Ukraine': 'Ukraine'
        },
        "B_COUNTRY_to_continent": {
            'I am currently in Africa': 'Africa',
            'I am currently in North America': 'North America',
            'I am currently in Europe': 'Europe',
            'I am currently in Asia': 'Asia',
            'I am currently in South America': 'South America',
            'I am currently in Oceania': 'Oceania'
        },
        "Q260": {
            'I am a male': 'Male',
            'I am a female': 'Female'
        },
        "X003R": {
            'I am 45-54 years old': '45-54',
            'I am 35-44 years old': '35-44',
            'I am 65+ years old': '65+',
            'I am 25-34 years old': '25-34',
            'I am 16-24 years old': '16-24',
            'I am 55-64 years old': '55-64'
        },
        "Q263": {
            'I was born in this country': 'Citizen',
            'I am an immigrant to this country': 'Immigrant'
        },
        "Q266": {
            'I was born in Libya': 'Libya',
            'I was born in Ethiopia': 'Ethiopia',
            'I was born in Slovakia': 'Slovakia',
            'I was born in Malaysia': 'Malaysia',
            'I was born in Zimbabwe': 'Zimbabwe',
            'I was born in Morocco': 'Morocco',
            'I was born in Canada': 'Canada',
            'I was born in Singapore': 'Singapore',
            'I was born in Kenya': 'Kenya',
            'I was born in Taiwan ROC': 'Taiwan ROC',
            'I was born in Chile': 'Chile',
            'I was born in South Korea': 'South Korea',
            'I was born in United States': 'United States',
            'I was born in Azerbaijan': 'Azerbaijan',
            'I was born in Trinidad and Tobago': 'Trinidad and Tobago',
            'I was born in Argentina': 'Argentina',
            'I was born in Indonesia': 'Indonesia',
            'I was born in Tunisia': 'Tunisia',
            'I was born in Puerto Rico': 'Puerto Rico',
            'I was born in Serbia': 'Serbia',
            'I was born in New Zealand': 'New Zealand',
            'I was born in Colombia': 'Colombia',
            'I was born in Russia': 'Russia',
            'I was born in Brazil': 'Brazil',
            'I was born in Philippines': 'Philippines',
            'I was born in Romania': 'Romania',
            'I was born in Mexico': 'Mexico',
            'I was born in Pakistan': 'Pakistan',
            'I was born in United Kingdom': 'United Kingdom',
            'I was born in Peru': 'Peru',
            'I was born in Bolivia': 'Bolivia',
            'I was born in Mongolia': 'Mongolia',
            'I was born in Ecuador': 'Ecuador',
            'I was born in Syria': 'Syria',
            'I was born in Nicaragua': 'Nicaragua',
            'I was born in Hong Kong SAR': 'Hong Kong SAR',
            'I was born in Cyprus': 'Cyprus',
            'I was born in Nigeria': 'Nigeria',
            'I was born in Bangladesh': 'Bangladesh',
            'I was born in Sri Lanka': 'Sri Lanka',
            'I was born in Guatemala': 'Guatemala',
            'I was born in Lebanon': 'Lebanon',
            'I was born in Greece': 'Greece',
            'I was born in North Macedonia': 'North Macedonia',
            'I was born in Thailand': 'Thailand',
            'I was born in Uruguay': 'Uruguay',
            'I was born in Armenia': 'Armenia',
            'I was born in Poland': 'Poland',
            'I was born in India': 'India',
            'I was born in Estonia': 'Estonia',
            'I was born in China': 'China',
            'I was born in Netherlands': 'Netherlands',
            'I was born in Vietnam': 'Vietnam',
            'I was born in Ukraine': 'Ukraine',
            'I was born in Germany': 'Germany',
            'I was born in Venezuela': 'Venezuela',
            'I was born in South Africa': 'South Africa',
            'I was born in Algeria': 'Algeria',
            'I was born in Panama': 'Panama',
            'I was born in Macau SAR': 'Macau SAR'
        },
        "Q269": {
            'I am a citizen of this country': 'Citizen',
            'I am not a citizen of this country': 'Not Citizen'
        },
        "Q273": {
            'I am married': 'Married',
            'I am widowed': 'Widowed',
            'I am single': 'Single',
            'I am separated': 'Separated',
            'I am living together as married': 'Unmarried Living Together',
            'I am divorced': 'Divorced'
        },
        "Q275": {
            'The highest educational level that I have attained is upper secondary education': 'Upper Secondary',
            'The highest educational level that I have attained is short-cycle tertiary education': 'Short-Cycle Tertiary',
            'The highest educational level that I have attained is primary education': 'Primary',
            'The highest educational level that I have attained is bachelor or equivalent': 'Bachelor or Equivalent',
            'The highest educational level that I have attained is lower secondary education': 'Lower Secondary',
            'The highest educational level that I have attained is doctoral or equivalent': 'Doctoral or Equivalent',
            'The highest educational level that I have attained is master or equivalent': 'Master or Equivalent',
            'The highest educational level that I have attained is post-secondary non-tertiary education': 'Post-Secondary Non-Tertiary',
            'The highest educational level that I have attained is early childhood education or no education': 'Early Childhood or None'
        },
        "Q279": {
            'I am employed full time': 'Employed Full Time',
            'I am retired or pensioned': 'Retired or Pensioned',
            'I am self employed': 'Self Employed',
            'I am unemployed': 'Unemployed',
            'I am employed part time': 'Employed Part Time',
            'I am a student': 'Student',
            'I am a housewife and not otherwise employed': 'Housewife'
        },
        "Q281": {
            'I have a professional and technical job, e.g., doctor, teacher, engineer, artist, accountant, nurse': 'Professional & Technical',
            'I have an unskilled worker job, e.g., labourer, porter, unskilled factory worker, cleaner': 'Unskilled Worker',
            'I have never had a job': 'No Job',
            'I have a clerical job, e.g., secretary, clerk, office manager, civil servant, bookkeeper': 'Clerical',
            'I have a service job, e.g., restaurant owner, police officer, waitress, barber, caretaker': 'Service',
            'I have a sales job, e.g., sales manager, shop owner, shop assistant, insurance agent, buyer': 'Sales',
            'I have a semi-skilled worker job, e.g., bricklayer, bus driver, cannery worker, carpenter, sheet metal worker, baker': 'Semi-Skilled Worker',
            'I have a higher administrative job, e.g., banker, executive in big business, high government official, union official': 'Higher Administrative',
            'I have a farm owner or farm manager job': 'Farm Owner',
            'I have a farm worker job, e.g., farm labourer, tractor driver': 'Farm Worker',
            'I have a skilled worker job, e.g., foreman, motor mechanic, printer, seamstress, tool and die maker, electrician': 'Skilled Worker'
        },
        "Q284": {
            'I am working for or have worked for government or public institution': 'Government or Public Institution',
            'I am working for or have worked for private business or industry': 'Private Business or Industry',
            'I am working for or have worked for private non-profit organization': 'Private Non-Profit Organization'
        },
        "Q286": {
            'during the past year, my family was not able to save money': 'Not Able to Save Money',
            'during the past year, my family was able to save money': 'Able to Save Money'
        },
        "Q287": {
            'I would describe myself as belonging to the working class': 'Working Class',
            'I would describe myself as belonging to the upper middle class': 'Upper Middle Class',
            'I would describe myself as belonging to the lower class': 'Lower Class',
            'I would describe myself as belonging to the lower middle class': 'Lower Middle Class',
            'I would describe myself as belonging to the upper class': 'Upper Class'
        },
        "Q288R": {
            'My household is among the middle income households in my country': 'Middle Income',
            'My household is among the low income households in my country': 'Low Income',
            'My household is among the high income households in my country': 'High Income'
        },
        "Q289": {
            'I belong to the Muslim religion': 'Muslim',
            'I belong to the Orthodox (Russian/Greek/etc.) religion': 'Orthodox',
            'I belong to the Roman Catholic religion': 'Catholic',
            'I belong to the Hindu religion': 'Hindu',
            'I belong to the Protestant religion': 'Protestant',
            'I belong to the Jew religion': 'Jewish',
            'I belong to some other Christian (Evangelical/Pentecostal/Fee church/etc.) religion': 'Other Christian',
            'I belong to no religion or religious denomination': 'No Religion',
            'I belong to some other religion or religious denomination': 'Other Religion',
            'I belong to the Buddhist religion': 'Buddhist'
        }
    }

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
                    all_acc[demographics_qid][demographics_dimension].extend(
                        acc_by_demographics_dimension[demographics_dimension])

    for demographics_qid in all_acc:
        print("=" * 30, demographics_qid, "=" * 30)
        print(list(all_acc[demographics_qid].keys()))

    # for demo_dim in acc_by_demographics_dimension:
    #     print(list(acc_by_demographics_dimension[demo_dim].keys()))

    sns.set_style("whitegrid")

    for i, (demographics_qid, acc_by_demographics_dimension) in enumerate(all_acc.items()):
        converted_statements = demographics_metadata[demographics_qid]["converted_statements"]

        plot_data = []
        for dimension, accuracies in acc_by_demographics_dimension.items():
            plot_data.extend([(dimension, acc) for acc in accuracies])

        # Create a DataFrame
        df = pd.DataFrame(plot_data, columns=[
                          'Demographic Dimension', 'Accuracy'])

        df["Accuracy"] = df["Accuracy"] * 100

        # Calculate the number of dimensions and set the figure size
        num_dimensions = len(df['Demographic Dimension'].unique())
        # Increased width to accommodate spacing
        fig_width = min(15, num_dimensions * 0.6)
        fig_height = 4

        # Create a new figure for each plot with adjusted width
        plt.figure(figsize=(fig_width, fig_height))

        # Create the box plot with width of 0.8 and increased spacing
        # Create a mapping of demographic dimensions to their order in converted_statements
        dimension_order = {dim: i for i,
                           dim in enumerate(converted_statements)}

        # Sort the DataFrame based on this order
        df['order'] = df['Demographic Dimension'].map(dimension_order)
        df = df.sort_values('order')

        ax = sns.boxplot(x='Demographic Dimension',
                         y='Accuracy',
                         data=df,
                         width=0.5,
                         native_scale=True,
                         color=sns.color_palette("Set2")[2],
                         order=df['Demographic Dimension'].unique())
        # Remove x-axis ticks
        plt.xticks([])

        # Add x-axis labels directly on the plot
        for i, label in enumerate(df['Demographic Dimension'].unique()):
            label = convert_stmt_to_short_form[demographics_qid][label]
            ax.text(i, ax.get_ylim()[0] - 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]) + 0.05,
                    label, ha='center', va='top', rotation=90, fontsize=12)

        # Remove x and y labels
        plt.xlabel('')
        plt.ylabel('')

        demographics_name = core_demographics_QIDs_to_demographics_name[demographics_qid]
        plt.xticks(rotation=90)
        plt.title(demographics_name, fontsize=14)

        plt.tight_layout()
        plot_filename = f"results/plots/box/{demographics_name}.png"
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        print(f"Box plot for {demographics_name} saved as {plot_filename}")

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="gpt-4o-2024-08-06")
    parser.add_argument("--num_stmts", type=int, default=200)
    parser.add_argument("--probe_setup_id", type=int, default=0)
    parser.add_argument("--data_version", type=str,
                        default="090824_800")
    parser.add_argument("--mode", type=str, default="stmt")
    args = parser.parse_args()

    main_compare_models()

    # plot_heatmap_broken_down_by_category()

    # main_demographics_number_of_statements(model_name="gpt-4o-mini-2024-07-18",
    #                                        data_version="090824_800")

    # plot_by_demographics_dimension_new(model_name=args.model_name,
    #                                    num_stmts=args.num_stmts,
    #                                    probe_setup_id=args.probe_setup_id,
    #                                    mode=args.mode,
    #                                    data_version=args.data_version)

    # main_demographics_number_of_statements(model_name=args.model_name,
    #                                        data_version=args.data_version)

    # plot_by_demographics_dimensions(model_name="gpt-4o-2024-08-06",
    #                                 num_stmts=200,
    #                                 mode="stmt",
    #                                 data_version="090824_800")

    # plot_heatmap_broken_down_by_category()
