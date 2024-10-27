import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from data.WVS.WVS_conversion import *


def load_probe_data(data_version, is_refined):
    if is_refined:
        data_path = f"data/WVS/human_labels/probe_data/{data_version}_refined.jsonl"
    else:
        data_path = f"data/WVS/human_labels/probe_data/{data_version}.jsonl"
    probe_data = load_standard_data(data_path, is_print=False)
    return {data_d["D_INTERVIEW"]: data_d for data_d in probe_data}


def get_accuracy(d, probe_qids):
    d_acc = []
    for qid in probe_qids:
        qid_is_correct = d[f"{qid}_is_correct"]
        d_acc.append(qid_is_correct)
    d_acc = np.mean(d_acc)
    return d_acc


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


def get_ref_all_acc(demographics_qid_to_plot, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    all_acc = {}
    for probe_setup_id in range(3):
        expt_id = f"stmt-090824_800-v{probe_setup_id}-num_stmt_200"
        data_path = f"data/WVS/probe_expts/090824_800/{expt_id}.jsonl"
        print(f"{model_name}/{expt_id}.jsonl")
        results_path = data_path.replace(
            "data/WVS/", "results/").replace(f"{expt_id}.jsonl", f"{model_name}/is_correct/{expt_id}.jsonl")
        results_data = load_standard_data(results_path, is_print=False)
        probe_qids = get_all_probe_qids(probe_setup_id=probe_setup_id)

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
    return all_acc


def plot_by_demographics_dimensions(model_name,
                                    data_version,
                                    num_demo,
                                    demo_mode,
                                    probe_mode,
                                    demographics_qid_to_plot=core_demographics_QIDs_to_demographics_name.keys()):
    # demographics_qid_to_plot=["B_COUNTRY", "Q266"]):

    # demographics_qid_to_plot=core_demographics_QIDs_to_demographics_name.keys()):
    # ["Q288R", "B_COUNTRY_to_continent", "Q287"]):

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

    demographics_metadata = get_demographics_metadata_map()

    all_acc = {qid: {} for qid in demographics_qid_to_plot}
    for probe_setup_id in range(3):
        demo_human_label_data, demo_metadata_map, probe_human_label_data_map, probe_metadata_map, all_probe_QIDs, split_probe_human_ids, all_statements_QIDs = load_data(
            demo_mode, probe_mode, probe_setup_id, data_version)

        for split in ["eval"]:  # "val", "test"
            raw_data_path = f"results/finetune_expts/{model_name}/{demo_mode}{num_demo}_{probe_mode}_{split}_{probe_setup_id}/raw_outputs.jsonl"
            model_gen_data = load_standard_data(raw_data_path, is_print=False)

            human_ids_data_path = f"data/WVS/training_expts/{split}/v1/human_ids.jsonl"
            human_ids_data = load_standard_data(
                human_ids_data_path, is_print=False)

            # double check the data corresponds to the correct human ids
            print("Checking data...")
            for model_gen_d, human_id_d in tqdm(zip(model_gen_data, human_ids_data), total=len(model_gen_data), desc="Checking data"):
                human_id = human_id_d["D_INTERVIEW"]
                demo_qids = model_gen_d["demo_qids"]
                demo_statements = model_gen_d["demo_statements"]

                for demo_qid, demo_statement in zip(demo_qids, demo_statements):
                    demo_gaid = demo_human_label_data[human_id][demo_qid]
                    demo_statement_converted = demo_metadata_map[demo_qid]["converted_statements"][demo_gaid]

                    if demo_statement_converted != demo_statement:
                        raise Exception(
                            f"Mismatch for {human_id} {demo_qid}: {demo_statement_converted} != {demo_statement}")
            print("Successfully checked data!")

            print("Sorting data...")
            accuracy_by_human_id = {}
            for model_gen_d, human_id_d in tqdm(zip(model_gen_data, human_ids_data), total=len(model_gen_data), desc="Sorting data"):
                human_id = human_id_d["D_INTERVIEW"]
                model_choice_id = model_gen_d["model_choice_id"]
                label_choice_id = model_gen_d["label_choice_id"]
                acc_d = int(model_choice_id == label_choice_id)

                if human_id not in accuracy_by_human_id:
                    accuracy_by_human_id[human_id] = []
                accuracy_by_human_id[human_id].append(acc_d)

            for human_id, accuracies in accuracy_by_human_id.items():
                avg_acc = np.mean(accuracies)
                for demographics_qid in demographics_qid_to_plot:
                    demographics_stmt = demo_human_label_data[human_id][demographics_qid]
                    if demographics_stmt == -99:
                        continue
                    if demographics_stmt not in all_acc[demographics_qid]:
                        all_acc[demographics_qid][demographics_stmt] = []
                    all_acc[demographics_qid][demographics_stmt].append(
                        avg_acc)
            print("Successfully sorted data!")

    print("=" * 10, "All Acc", "=" * 10)
    for demographics_qid in demographics_qid_to_plot:
        print("=" * 10, demographics_qid, "=" * 10)
        for d_s in all_acc[demographics_qid]:
            print(d_s, np.mean(all_acc[demographics_qid][d_s]))

    ref_all_acc = get_ref_all_acc(demographics_qid_to_plot)
    print("=" * 10, "Ref All Acc", "=" * 10)
    for demographics_qid in demographics_qid_to_plot:
        print("=" * 10, demographics_qid, "=" * 10)
        for d_s in ref_all_acc[demographics_qid]:
            print(d_s, np.mean(ref_all_acc[demographics_qid][d_s]))

    ##########################################################################################

    sns.set_style("whitegrid")

    for i, (demographics_qid, acc_by_demographics_dimension) in enumerate(all_acc.items()):
        converted_statements = demographics_metadata[demographics_qid]["converted_statements"]

        plot_data = []
        acc_by_demographics_dimension_ref = ref_all_acc[demographics_qid]
        for dimension, accuracies in acc_by_demographics_dimension_ref.items():
            plot_data.extend([(dimension, acc, "Zero-Shot")
                             for acc in accuracies])

        for dimension, accuracies in acc_by_demographics_dimension.items():
            plot_data.extend([(dimension, acc, "Reasoner")
                             for acc in accuracies])

        # Create a DataFrame
        df = pd.DataFrame(plot_data, columns=[
                          'Demographic Dimension', 'Accuracy', 'Model'])
        df["Accuracy"] = df["Accuracy"] * 100
        # Calculate the number of dimensions and set the figure size
        num_dimensions = len(df['Demographic Dimension'].unique())
        # Increased width to accommodate spacing
        # fig_width = min(40, num_dimensions * 0.85)
        # fig_height = 4.8

        fig_width = min(40, num_dimensions * 1)
        fig_height = 5.5

        # Create a new figure for each plot with adjusted width
        plt.figure(figsize=(fig_width, fig_height))

        # Create the box plot with width of 0.8 and increased spacing
        # Create a mapping of demographic dimensions to their order in converted_statements
        dimension_order = {dim: i for i,
                           dim in enumerate(converted_statements)}

        # Sort the DataFrame based on this order
        df['order'] = df['Demographic Dimension'].map(dimension_order)
        df = df.sort_values('order')

        # Order by ['Zero-shot', 'Reasoner']
        df['Model'] = pd.Categorical(df['Model'], categories=[
                                     'Zero-Shot', 'Reasoner'], ordered=True)
        df = df.sort_values(['order', 'Model'])
        demographic_dimensions_in_plot_order = df['Demographic Dimension'].unique(
        )

        # Calculate mean differences
        mean_diff = df[df['Model'] == 'Reasoner'].groupby('Demographic Dimension')['Accuracy'].mean(
        ) - df[df['Model'] == 'Zero-Shot'].groupby('Demographic Dimension')['Accuracy'].mean()
        mean_diff_dict = mean_diff.to_dict()
        mean_diff = [(dim, mean_diff_dict[dim])
                     for dim in demographic_dimensions_in_plot_order]

        df_by_cat = df.groupby(['Demographic Dimension', 'Model'])[
            'Accuracy'].mean()
        variances = df_by_cat.groupby('Model').std()
        variances_dict = variances.to_dict()
        mean = df_by_cat.groupby('Model').mean()
        mean_dict = mean.to_dict()

        print(
            "=" * 10, core_demographics_QIDs_to_demographics_name[demographics_qid], "=" * 10)
        print("Std:", variances_dict)
        print("Mean:", mean_dict)

        ax = sns.boxplot(x='Demographic Dimension',
                         y='Accuracy',
                         hue='Model',
                         data=df,
                         width=0.5,
                         native_scale=True,
                         palette=[sns.color_palette(
                             "Set2")[2], sns.color_palette("Set2")[3]],
                         order=df['Demographic Dimension'].unique())
        # Remove x-axis ticks
        plt.xticks([])

        # Add x-axis labels directly on the plot
        for i, label in enumerate(df['Demographic Dimension'].unique()):
            label = convert_stmt_to_short_form[demographics_qid][label]
            ax.text(i, ax.get_ylim()[0] - 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]) + 0.05,
                    label, ha='center', va='top', rotation=90, fontsize=11)

        # Add labels for mean differences
        for i, (dimension, diff) in enumerate(mean_diff):
            ax.text(i, ax.get_ylim()[1],
                    f'+{diff:.2f}', ha='center', va='bottom')

        # Remove x and y labels
        plt.xlabel('')
        plt.ylabel('')

        demographics_name = core_demographics_QIDs_to_demographics_name[demographics_qid]
        plt.xticks(rotation=90)
        plt.title(
            f"{demographics_name} (σ={variances_dict['Zero-Shot']:.2f} vs. {variances_dict['Reasoner']:.2f})", fontsize=12, y=1.06)
        # Remove the legend
        ax.get_legend().remove()

        # # Move the legend outside of the plot
        # plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

        # # Adjust the figure size to accommodate the legend
        # fig = plt.gcf()
        # fig.set_size_inches(fig.get_size_inches()[0] * 1.2, fig.get_size_inches()[1])

        plt.tight_layout()
        plot_filename = f"results/plots/finetune/box/{model_name}/{demographics_name}.png"
        plt.savefig(plot_filename, bbox_inches='tight', dpi=500)
        print(f"Box plot for {demographics_name} saved as {plot_filename}")

        plt.close()


def plot_by_continent_models(data_version,
                             num_demo,
                             demo_mode,
                             probe_mode,
                             is_check_data=False):

    demographics_qid_to_plot = ["B_COUNTRY_to_continent"]

    list_of_continents = ["Africa", "Europe", "North America",
                          "Oceania", "South America", "Asia", "All"]

    # all_acc = {qid: {} for qid in demographics_qid_to_plot}
    all_acc_by_continent = {continent: {
        qid: {} for qid in demographics_qid_to_plot} for continent in list_of_continents}
    for model_continent in tqdm(list_of_continents, desc="Continents"):
        model_continent_no_blank = model_continent.replace(" ", "_")
        if model_continent == "All":
            model_name = f"p200_p1-r200_p1-p200_r1-r200_r1-n50_full"
        else:
            model_name = f"p200_p1-r200_p1-p200_r1-r200_r1-n50_full-{model_continent_no_blank}"

        for probe_setup_id in range(3):
            demo_human_label_data, demo_metadata_map, probe_human_label_data_map, probe_metadata_map, all_probe_QIDs, split_probe_human_ids, all_statements_QIDs = load_data(
                demo_mode, probe_mode, probe_setup_id, data_version)
            for split in ["val", "test"]:
                raw_data_path = f"results/finetune_expts/{model_name}/{demo_mode}{num_demo}_{probe_mode}_{split}_{probe_setup_id}/raw_outputs.jsonl"
                model_gen_data = load_standard_data(
                    raw_data_path, is_print=False)

                human_ids_data_path = f"data/WVS/training_expts/{split}/v1/human_ids.jsonl"
                human_ids_data = load_standard_data(
                    human_ids_data_path, is_print=False)

                if is_check_data:
                    # double check the data corresponds to the correct human ids
                    print("Checking data...")
                    for model_gen_d, human_id_d in tqdm(zip(model_gen_data, human_ids_data), total=len(model_gen_data), desc="Checking data"):
                        human_id = human_id_d["D_INTERVIEW"]
                        demo_qids = model_gen_d["demo_qids"]
                        demo_statements = model_gen_d["demo_statements"]

                        for demo_qid, demo_statement in zip(demo_qids, demo_statements):
                            demo_gaid = demo_human_label_data[human_id][demo_qid]
                            demo_statement_converted = demo_metadata_map[
                                demo_qid]["converted_statements"][demo_gaid]

                            if demo_statement_converted != demo_statement:
                                raise Exception(
                                    f"Mismatch for {human_id} {demo_qid}: {demo_statement_converted} != {demo_statement}")
                    print("Successfully checked data!")

                print("Sorting data...")
                accuracy_by_human_id = {}
                for model_gen_d, human_id_d in tqdm(zip(model_gen_data, human_ids_data), total=len(model_gen_data), desc="Sorting data"):
                    human_id = human_id_d["D_INTERVIEW"]
                    model_choice_id = model_gen_d["model_choice_id"]
                    label_choice_id = model_gen_d["label_choice_id"]
                    acc_d = int(model_choice_id == label_choice_id)

                    if human_id not in accuracy_by_human_id:
                        accuracy_by_human_id[human_id] = []
                    accuracy_by_human_id[human_id].append(acc_d)

                for human_id, accuracies in accuracy_by_human_id.items():
                    avg_acc = np.mean(accuracies)
                    for demographics_qid in demographics_qid_to_plot:
                        demographics_stmt = demo_human_label_data[human_id][demographics_qid]
                        if demographics_stmt == -99:
                            continue
                        if demographics_stmt not in all_acc_by_continent[model_continent][demographics_qid]:
                            all_acc_by_continent[model_continent][demographics_qid][demographics_stmt] = [
                            ]
                        all_acc_by_continent[model_continent][demographics_qid][demographics_stmt].append(
                            avg_acc)
                print("Successfully sorted data!")

    ################## add the all column ##################
    for demographics_qid in demographics_qid_to_plot:
        for continent in list_of_continents:
            d_s_all = []
            for d_s in all_acc_by_continent[continent][demographics_qid]:
                d_s_all.extend(
                    all_acc_by_continent[continent][demographics_qid][d_s])
            all_acc_by_continent[continent][demographics_qid]["All"] = d_s_all

    ###########################################################

    # Prepare data for heatmap
    heatmap_data = {}
    for continent in list_of_continents:
        heatmap_data[continent] = {}
        for demographics_qid in demographics_qid_to_plot:
            for d_s_continent in list_of_continents:
                if d_s_continent == "All":
                    d_s = "All"
                else:
                    d_s = "I am currently in " + d_s_continent
                heatmap_data[continent][f"{d_s_continent}"] = np.mean(
                    all_acc_by_continent[continent][demographics_qid][d_s]) * 100

    # Convert to DataFrame
    df = pd.DataFrame(heatmap_data).T

    # Create heatmap
    plt.figure(figsize=(7, 4.4))
    sns.heatmap(df, annot=True, cmap="YlGnBu",
                fmt=".2f", annot_kws={"size": 11})
    # plt.title("Accuracy by Continent and Demographic Dimension", fontsize=18)
    plt.xlabel("Evaluation Population", fontsize=13)
    plt.ylabel("Model", fontsize=13)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(
        f"results/plots/finetune/heatmap/heatmap_by_continent.png", dpi=600)
    plt.close()

    print(f"Heatmap saved as results/finetune_expts/heatmap/heatmap_by_continent.png")


def plot_by_num_demo(data_version,
                     demo_mode,
                     probe_mode,
                     is_check_data=False,
                     model_names=[
                         #  "p200_p1-r200_p1-p200_r1-r200_r1-prand_p1-rrand_p1-prand_r1-rrand_r1-n50_full",
                         #  "p200_p1-r200_p1-p200_r1-r200_r1-prand_p1-rrand_p1-prand_r1-rrand_r1-n100_full",
                         #  "p200_p1-r200_p1-p200_r1-r200_r1-prand_p1-rrand_p1-prand_r1-rrand_r1-n150_full",
                         #  "p200_p1-r200_p1-p200_r1-r200_r1-prand_p1-rrand_p1-prand_r1-rrand_r1-n200_full",
                         #  "p200_p1-r200_p1-p200_r1-r200_r1-n200_full",
                         #  "p150_p1-r150_p1-p150_r1-r150_r1-n200_full",
                         #  "p100_p1-r100_p1-p100_r1-r100_r1-n200_full",
                         #  "p50_p1-r50_p1-p50_r1-r50_r1-n200_full"

                         #  "p200_p1-r200_p1-p200_r1-r200_r1-prand_p1-rrand_p1-prand_r1-rrand_r1-n100_full",
                         "all_mixed-all_200-n100",
                         "pmixed_p1-rmixed_p1-pmixed_r1-rmixed_r1-n200_full",
                         "p200_p1-r200_p1-p200_r1-r200_r1-n200_full",
                         "p150_p1-r150_p1-p150_r1-r150_r1-n200_full",
                         "p100_p1-r100_p1-p100_r1-r100_r1-n200_full",
                         "p50_p1-r50_p1-p50_r1-r50_r1-n200_full",
                     ],
                     is_reload_saved_data=True):

    compiled_save_path = f"results/plots/finetune/line/num_demo_vs_acc_new_clean.json"
    if not is_reload_saved_data:
        all_acc_by_num_demo = {model_name: {num_demo: []
                                            for num_demo in [25, 50, 75, 100, 125, 150, 175, 200]} for model_name in model_names}
        for model_name in model_names:
            for num_demo in tqdm(all_acc_by_num_demo[model_name], desc="Num Demo"):
                for probe_setup_id in range(3):
                    demo_human_label_data, demo_metadata_map, probe_human_label_data_map, probe_metadata_map, all_probe_QIDs, split_probe_human_ids, all_statements_QIDs = load_data(
                        demo_mode, probe_mode, probe_setup_id, data_version)
                    for split in ["eval"]:
                        raw_data_path = f"results/finetune_expts/{model_name}/{demo_mode}{num_demo}_{probe_mode}_{split}_{probe_setup_id}/raw_outputs.jsonl"
                        model_gen_data = load_standard_data(
                            raw_data_path, is_print=False)

                        human_ids_data_path = f"data/WVS/training_expts/{split}/v1/human_ids.jsonl"
                        human_ids_data = load_standard_data(
                            human_ids_data_path, is_print=False)

                        if is_check_data:
                            # double check the data corresponds to the correct human ids
                            print("Checking data...")
                            for model_gen_d, human_id_d in tqdm(zip(model_gen_data, human_ids_data), total=len(model_gen_data), desc="Checking data"):
                                human_id = human_id_d["D_INTERVIEW"]
                                demo_qids = model_gen_d["demo_qids"]
                                demo_statements = model_gen_d["demo_statements"]

                                for demo_qid, demo_statement in zip(demo_qids, demo_statements):
                                    demo_gaid = demo_human_label_data[human_id][demo_qid]
                                    demo_statement_converted = demo_metadata_map[
                                        demo_qid]["converted_statements"][demo_gaid]

                                    if demo_statement_converted != demo_statement:
                                        raise Exception(
                                            f"Mismatch for {human_id} {demo_qid}: {demo_statement_converted} != {demo_statement}")
                            print("Successfully checked data!")

                        print("Sorting data...")
                        accuracy_by_human_id = {}
                        for model_gen_d, human_id_d in tqdm(zip(model_gen_data, human_ids_data), total=len(model_gen_data), desc="Sorting data"):
                            human_id = human_id_d["D_INTERVIEW"]
                            model_choice_id = model_gen_d["model_choice_id"]
                            label_choice_id = model_gen_d["label_choice_id"]
                            acc_d = int(model_choice_id == label_choice_id)

                            if human_id not in accuracy_by_human_id:
                                accuracy_by_human_id[human_id] = []
                            accuracy_by_human_id[human_id].append(acc_d)

                        for human_id, accuracies in accuracy_by_human_id.items():
                            avg_acc = np.mean(accuracies)
                            all_acc_by_num_demo[model_name][num_demo].append(
                                avg_acc)
                        print("Successfully sorted data!")

        with open(compiled_save_path, "w") as f:
            json.dump(all_acc_by_num_demo, f)

    else:
        with open(compiled_save_path, "r") as f:
            all_acc_by_num_demo = json.load(f)

    # Prepare data for plotting
    data = []
    for model_name in model_names:
        for num_demo, accuracies in all_acc_by_num_demo[model_name].items():
            mean_accuracy = np.mean(accuracies)

            # 95% confidence interval
            ci = 1.96 * np.std(accuracies) / np.sqrt(len(accuracies))
            data.append({
                'Number of Demonstration Stmt': int(num_demo),
                'Accuracy': mean_accuracy,
                'CI_lower': mean_accuracy - ci,
                'CI_upper': mean_accuracy + ci,
                'Model': model_name
            })

    df = pd.DataFrame(data)

    # remove the fig frame
    sns.set_style("whitegrid")

    # Create the plot using seaborn
    # plt.figure(figsize=(3.35, 3.5))
    # plt.figure(figsize=(3.25, 3.45))
    # plt.figure(figsize=(6, 6))
    plt.figure(figsize=(2.8, 3.6))
    # plt.figure(figsize=(3, 3.7))

    # colors = list(sns.color_palette("hls", 8))
    colors = list(sns.color_palette("Set2"))
    random.shuffle(colors)

    for model in model_names:
        model_data = df[df['Model'] == model]
        plt.plot(model_data['Number of Demonstration Stmt'],
                 model_data['Accuracy'],
                 marker='.',
                 label=model,
                 color=colors[model_names.index(model)])
        # plt.fill_between(model_data['Number of Demonstration Stmt'],
        #                  model_data['CI_lower'],
        #                  model_data['CI_upper'],
        #                  alpha=0.2,
        #                  color=colors[model_names.index(model)])

    # Customize the plot
    plt.xlabel('Num of Value Statements')
    plt.ylabel('Accuracy')
    # plt.title('Accuracy vs Number of Demonstration Statements by Model')
    plt.grid(True, linestyle='-', alpha=0.7)

    # Set x-axis ticks
    plt.xticks([25, 50, 75, 100, 125, 150, 175, 200], fontsize=8)

    plt.ylim(0.66, 0.747)

    # Add legend outside the plot frame
    plt.legend(title='Model', loc='center left', bbox_to_anchor=(1, 0.5))

    # remove legend
    # plt.legend().remove()

    # Save the plot
    plt.tight_layout()
    save_path = compiled_save_path.replace("_clean.json", ".png")
    plt.savefig(save_path,
                dpi=700, bbox_inches='tight')
    plt.close()

    print(f"Plot saved as {save_path}")


def plot_scaling_effect(data_version,
                        demo_mode,
                        probe_mode,
                        is_check_data=False,
                        model_names=[
                            "all_mixed-all_200-n200",
                            "all_mixed-all_200-n100",
                            "all_mixed-all_200-n75",
                            "all_mixed-all_200-n50",
                            "all_mixed-all_200-n25",
                        ],
                        is_reload_saved_data=True):

    compiled_save_path = f"results/plots/finetune/line/scaling_effect.json"
    if not is_reload_saved_data:
        all_acc_by_num_demo = {model_name: {num_demo: []
                                            for num_demo in [25, 50, 75, 100, 125, 150, 175, 200]} for model_name in model_names}
        for model_name in model_names:
            for num_demo in tqdm(all_acc_by_num_demo[model_name], desc="Num Demo"):
                for probe_setup_id in range(3):
                    demo_human_label_data, demo_metadata_map, probe_human_label_data_map, probe_metadata_map, all_probe_QIDs, split_probe_human_ids, all_statements_QIDs = load_data(
                        demo_mode, probe_mode, probe_setup_id, data_version)
                    for split in ["eval"]:
                        raw_data_path = f"results/finetune_expts/{model_name}/{demo_mode}{num_demo}_{probe_mode}_{split}_{probe_setup_id}/raw_outputs.jsonl"
                        model_gen_data = load_standard_data(
                            raw_data_path, is_print=False)

                        human_ids_data_path = f"data/WVS/training_expts/{split}/v1/human_ids.jsonl"
                        human_ids_data = load_standard_data(
                            human_ids_data_path, is_print=False)

                        if is_check_data:
                            # double check the data corresponds to the correct human ids
                            print("Checking data...")
                            for model_gen_d, human_id_d in tqdm(zip(model_gen_data, human_ids_data), total=len(model_gen_data), desc="Checking data"):
                                human_id = human_id_d["D_INTERVIEW"]
                                demo_qids = model_gen_d["demo_qids"]
                                demo_statements = model_gen_d["demo_statements"]

                                for demo_qid, demo_statement in zip(demo_qids, demo_statements):
                                    demo_gaid = demo_human_label_data[human_id][demo_qid]
                                    demo_statement_converted = demo_metadata_map[
                                        demo_qid]["converted_statements"][demo_gaid]

                                    if demo_statement_converted != demo_statement:
                                        raise Exception(
                                            f"Mismatch for {human_id} {demo_qid}: {demo_statement_converted} != {demo_statement}")
                            print("Successfully checked data!")

                        print("Sorting data...")
                        accuracy_by_human_id = {}
                        for model_gen_d, human_id_d in tqdm(zip(model_gen_data, human_ids_data), total=len(model_gen_data), desc="Sorting data"):
                            human_id = human_id_d["D_INTERVIEW"]
                            model_choice_id = model_gen_d["model_choice_id"]
                            label_choice_id = model_gen_d["label_choice_id"]
                            acc_d = int(model_choice_id == label_choice_id)

                            if human_id not in accuracy_by_human_id:
                                accuracy_by_human_id[human_id] = []
                            accuracy_by_human_id[human_id].append(acc_d)

                        for human_id, accuracies in accuracy_by_human_id.items():
                            avg_acc = np.mean(accuracies)
                            all_acc_by_num_demo[model_name][num_demo].append(
                                avg_acc)
                        print("Successfully sorted data!")

        with open(compiled_save_path, "w") as f:
            json.dump(all_acc_by_num_demo, f)

    else:
        with open(compiled_save_path, "r") as f:
            all_acc_by_num_demo = json.load(f)

    # Prepare data for plotting
    data = []
    for model_name in model_names:
        for num_demo, accuracies in all_acc_by_num_demo[model_name].items():
            mean_accuracy = np.mean(accuracies)

            # 95% confidence interval
            ci = 1.96 * np.std(accuracies) / np.sqrt(len(accuracies))
            data.append({
                'Number of Demonstration Stmt': int(num_demo),
                'Accuracy': mean_accuracy,
                'CI_lower': mean_accuracy - ci,
                'CI_upper': mean_accuracy + ci,
                'Model': model_name
            })

    df = pd.DataFrame(data)

    # remove the fig frame
    sns.set_style("whitegrid")

    # Create the plot using seaborn
    # plt.figure(figsize=(3.35, 3.5))
    # plt.figure(figsize=(3.25, 3.45))
    # plt.figure(figsize=(6, 6))
    plt.figure(figsize=(2.8, 3.6))
    # plt.figure(figsize=(3, 3.7))

    # colors = list(sns.color_palette("hls", 8))
    palette = list(sns.color_palette("Set2"))
    # random.shuffle(colors)

    colors = [palette[1], palette[2], palette[5], palette[3], palette[0]]

    for model in model_names:
        model_data = df[df['Model'] == model]
        plt.plot(model_data['Number of Demonstration Stmt'],
                 model_data['Accuracy'],
                 marker='o',
                 label=model,
                 color=colors[model_names.index(model)])
        plt.fill_between(model_data['Number of Demonstration Stmt'],
                         model_data['CI_lower'],
                         model_data['CI_upper'],
                         alpha=0.2,
                         color=colors[model_names.index(model)])

    # Customize the plot
    plt.xlabel('Num of Value Statements')
    plt.ylabel('Accuracy')
    # plt.title('Accuracy vs Number of Demonstration Statements by Model')
    plt.grid(True, linestyle='-', alpha=0.7)

    # Set x-axis ticks
    plt.xticks([25, 50, 75, 100, 125, 150, 175, 200], fontsize=8)

    plt.ylim(0.66, 0.752)

    # Add legend outside the plot frame
    plt.legend(title='Model', loc='center left', bbox_to_anchor=(1, 0.5))

    # remove legend
    plt.legend().remove()

    # Save the plot
    plt.tight_layout()
    save_path = compiled_save_path.replace(".json", "_clean.png")
    plt.savefig(save_path,
                dpi=700, bbox_inches='tight')
    plt.close()

    print(f"Plot saved as {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="all_mixed-all_200-n200")
    parser.add_argument("--data_version", type=str, default="090824_800")
    parser.add_argument("--num_demo", type=int, default=200)
    parser.add_argument("--demo_mode", type=str, default="polar")
    parser.add_argument("--probe_mode", type=str, default="polar")
    args = parser.parse_args()

    # plot_by_demographics_dimensions(model_name=args.model_name,
    #                                 data_version=args.data_version,
    #                                 num_demo=args.num_demo,
    #                                 demo_mode=args.demo_mode,
    #                                 probe_mode=args.probe_mode)

    # plot_by_continent_models(data_version=args.data_version,
    #                          num_demo=args.num_demo,
    #                          demo_mode=args.demo_mode,
    #                          probe_mode=args.probe_mode)

    plot_scaling_effect(data_version=args.data_version,
                        demo_mode=args.demo_mode,
                        probe_mode=args.probe_mode)
