from src.utils.main_utils import *
from data.WVS.WVS_conversion import *
import numpy as np
import time
from transformers import AutoTokenizer
# from utils.main_utils import *
from sentence_transformers import SentenceTransformer


def load_data(is_refined=False):
    probe_human_ids = get_probe_human_ids()
    if is_refined:
        base_human_label_data_path = "data/WVS/human_labels/demographics_in_nl_refined_statements_combined_full_set.jsonl"
    else:
        base_human_label_data_path = "data/WVS/human_labels/demographics_in_nl_statements_combined_full_set.jsonl"
    human_label_data = load_standard_data(
        base_human_label_data_path, is_print=False)
    human_label_data_map = {d["D_INTERVIEW"]: d for d in human_label_data}
    human_label_data_train = [
        d for d in human_label_data if d["D_INTERVIEW"] not in probe_human_ids]
    human_label_data_test_map = {
        d["D_INTERVIEW"]: d for d in human_label_data if d["D_INTERVIEW"] in probe_human_ids}

    if is_refined:
        statements_meta_data = load_standard_data(
            "data/WVS/meta_data/refined_statements_meta_data.jsonl", is_print=False)
        statements_meta_data_map = {
            d["question_id"]: d for d in statements_meta_data}
    else:
        statements_meta_data = load_standard_data(
            "data/WVS/meta_data/statements_meta_data.jsonl", is_print=False)
        statements_meta_data_map = {
            d["question_id"]: d for d in statements_meta_data}

    return probe_human_ids, human_label_data_train, human_label_data_test_map, statements_meta_data_map, human_label_data_map


def get_train_qids(probe_qids):
    all_qids = get_all_statements_QIDs()
    return [qid for qid in all_qids if qid not in probe_qids]


def get_probe_human_ids():
    data_file_path = "data/WVS/training_expts/eval/v1/human_ids.jsonl"
    data = load_standard_data(data_file_path, is_print=False)
    return list(set([d["D_INTERVIEW"] for d in data]))


def get_majority_vote_for_qid(qid, human_label_data_train):
    qid_gaids = [d[qid] for d in human_label_data_train]
    return max(set(qid_gaids), key=qid_gaids.count)


def compute_majority_vote_baseline(is_refined=False):
    probe_human_ids, human_label_data_train, human_label_data_test_map, statements_meta_data_map, human_label_data_map = load_data(
        is_refined)

    for probe_setup_id in range(3):
        print(f"========== Probe Setup {probe_setup_id} ==========")
        probe_qids = get_all_probe_qids(probe_setup_id)
        majority_vote_gaid_map = {}
        for qid in tqdm(probe_qids):
            majority_vote_gaid_map[qid] = get_majority_vote_for_qid(
                qid, human_label_data_train)

        all_accs = []
        for human_id in tqdm(probe_human_ids, desc=f"Computing accuracy..."):
            for qid in probe_qids:
                human_gaid = human_label_data_test_map[human_id][qid]
                majority_vote_gaid = majority_vote_gaid_map[qid]
                acc = int(human_gaid == majority_vote_gaid)
                all_accs.append(acc)

        print(
            f"Probe Setup {probe_setup_id}: {sum(all_accs) / len(all_accs)}", len(all_accs))


def prepare_resemblance_top_1_matrices_baseline(is_refined=False, top_sim_to_save=500):
    probe_human_ids, human_label_data_train, human_label_data_test_map, statements_meta_data_map, human_label_data_map = load_data(
        is_refined)

    import torch
    import csv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for probe_setup_id in range(3):
        print(f"========== Probe Setup {probe_setup_id} ==========")
        probe_qids = get_all_probe_qids(probe_setup_id)
        train_qids = get_train_qids(probe_qids)

        placeholder_matrix = []
        for t_qid in train_qids:
            qid_total_gaids = statements_meta_data_map[t_qid]["grouped_answer_ids"]
            placeholder_matrix.append([0.0 for _ in qid_total_gaids])

        # create a matrix and pad 0 to the longest list in the matrix
        max_length = max([len(row) for row in placeholder_matrix])
        for row in placeholder_matrix:
            row.extend([0.0] * (max_length - len(row)))
        placeholder_matrix = torch.tensor(
            placeholder_matrix, dtype=torch.float32, device=device)

        # get all human label matrices
        all_human_label_matrices_train = []
        all_human_label_matrices_train_human_ids = []
        all_human_label_matrices_map = {}

        # Pre-allocate torch tensors for faster operations
        human_label_matrix = torch.zeros_like(
            placeholder_matrix, dtype=torch.float32, device=device)

        # random_human_ids = random.sample(list(human_label_data_map.keys()), 2000) + probe_human_ids
        random_human_ids = list(human_label_data_map.keys())
        for human_id in tqdm(random_human_ids, desc="Processing human label data"):
            human_data = human_label_data_map[human_id]
            human_label_matrix.copy_(placeholder_matrix)
            for i, t_qid in enumerate(train_qids):
                t_gaid = human_data[t_qid]
                if t_gaid != -99:
                    human_label_matrix[i, t_gaid] = 1.0
            if human_id not in probe_human_ids:
                all_human_label_matrices_train.append(
                    human_label_matrix.clone())
                all_human_label_matrices_train_human_ids.append(human_id)
            all_human_label_matrices_map[human_id] = human_label_matrix.clone()

        all_human_label_matrices_train = torch.stack(
            all_human_label_matrices_train)

        print("Finished loading human label matrices.")
        print("Starting to load probe matrices.")
        probe_matrices = torch.stack(
            [all_human_label_matrices_map[id] for id in probe_human_ids])
        print(all_human_label_matrices_train.shape)
        print(probe_matrices.shape)

        # get time
        start_time = time.time()
        print(f"Starting similarity computation...")

        # Batchify the probe matrices
        batch_size = 2000  # Adjust this based on your GPU memory constraints
        similarities = []
        for i in range(0, probe_matrices.shape[0], batch_size):
            batch = probe_matrices[i:i+batch_size]
            all_human_label_matrices_train_flat = all_human_label_matrices_train.view(
                all_human_label_matrices_train.shape[0], -1)  # Reshape to [97, 214 * 4]
            # Reshape to [800, 214 * 4]
            batch_flat = batch.view(batch.shape[0], -1)

            batch_similarities = torch.matmul(
                all_human_label_matrices_train_flat, batch_flat.T)
            similarities.append(batch_similarities)

            end_time = time.time()
            print(f"i: {i}, Time taken: {end_time - start_time} seconds")

        similarities = torch.cat(similarities, dim=1)

        all_data_to_save = []
        for probe_idx, probe_human_id in enumerate(probe_human_ids):
            probe_human_d = {"probe_human_id": probe_human_id}
            top_indices = torch.argsort(similarities[:, probe_idx], descending=True)[
                :top_sim_to_save]
            probe_human_d["top_similarity_human_ids"] = [
                all_human_label_matrices_train_human_ids[i] for i in top_indices.cpu().numpy()]
            probe_human_d["top_similarities"] = similarities[top_indices,
                                                             probe_idx].cpu().numpy().tolist()
            all_data_to_save.append(probe_human_d)

        if is_refined:
            write_standard_data(
                all_data_to_save, f"results/baselines/resemblance_top_1_matrices/refined_probe_setup_{probe_setup_id}_top_{top_sim_to_save}.jsonl")
        else:
            write_standard_data(
                all_data_to_save, f"results/baselines/resemblance_top_1_matrices/probe_setup_{probe_setup_id}_top_{top_sim_to_save}.jsonl")


def compute_resemblance_top_1_matrices_baseline(is_refined=False, top_sim_to_save=500):
    probe_human_ids, human_label_data_train, human_label_data_test_map, statements_meta_data_map, human_label_data_map = load_data(
        is_refined)
    all_accs = []
    for probe_setup_id in range(3):
        if is_refined:
            data_to_load = f"results/baselines/resemblance_top_1_matrices/refined_probe_setup_{probe_setup_id}_top_{top_sim_to_save}.jsonl"
        else:
            data_to_load = f"results/baselines/resemblance_top_1_matrices/probe_setup_{probe_setup_id}_top_{top_sim_to_save}.jsonl"
        data = load_standard_data(data_to_load)

        for sim_d in data:
            probe_human_id = sim_d["probe_human_id"]
            top_similarity_human_ids = sim_d["top_similarity_human_ids"]
            top_sim_human_id = top_similarity_human_ids[0]

            for probe_qid in get_all_probe_qids(probe_setup_id):
                probe_human_label = human_label_data_map[probe_human_id][probe_qid]
                top_sim_human_label = human_label_data_map[top_sim_human_id][probe_qid]
                if probe_human_label == top_sim_human_label:
                    all_accs.append(1)
                else:
                    all_accs.append(0)

        print(f"Probe Setup {probe_setup_id}: {sum(all_accs) / len(all_accs)}")


def compute_resemblance_top_k_matrices_baseline(is_refined=False, top_sim_to_save=500, top_k=10):
    probe_human_ids, human_label_data_train, human_label_data_test_map, statements_meta_data_map, human_label_data_map = load_data(
        is_refined)
    all_accs = []
    for probe_setup_id in range(3):
        if is_refined:
            data_to_load = f"results/baselines/resemblance_top_1_matrices/refined_probe_setup_{probe_setup_id}_top_{top_sim_to_save}.jsonl"
        else:
            data_to_load = f"results/baselines/resemblance_top_1_matrices/probe_setup_{probe_setup_id}_top_{top_sim_to_save}.jsonl"
        data = load_standard_data(data_to_load, is_print=False)

        print(f"========== Probe Setup {probe_setup_id} ==========")
        for top_k in range(1, top_k + 1):
            all_accs = []
            for sim_d in data:
                probe_human_id = sim_d["probe_human_id"]
                top_similarity_human_ids = sim_d["top_similarity_human_ids"]
                top_k_sim_human_id = top_similarity_human_ids[:top_k]

                for probe_qid in get_all_probe_qids(probe_setup_id):
                    probe_human_label = human_label_data_map[probe_human_id][probe_qid]
                    top_k_sim_human_labels = [
                        human_label_data_map[top_sim_human_id][probe_qid] for top_sim_human_id in top_k_sim_human_id]
                    top_k_sim_human_label = max(
                        set(top_k_sim_human_labels), key=top_k_sim_human_labels.count)
                    if probe_human_label == top_k_sim_human_label:
                        all_accs.append(1)
                    else:
                        all_accs.append(0)

            # print(f"Probe Setup {probe_setup_id} top_k={top_k}: {sum(all_accs) / len(all_accs)}")
            print(sum(all_accs) / len(all_accs))


def prepare_resemblance_top_1_count_baseline(is_refined=False, top_sim_to_save=500):
    probe_human_ids, human_label_data_train, human_label_data_test_map, statements_meta_data_map, human_label_data_map = load_data(
        is_refined)

    for probe_setup_id in range(3):
        print(f"========== Probe Setup {probe_setup_id} ==========")
        probe_qids = get_all_probe_qids(probe_setup_id)
        train_qids = get_train_qids(probe_qids)

        all_data_to_save = []
        for probe_human_id in tqdm(probe_human_ids, desc="Processing probe human data"):
            probe_human_d = human_label_data_map[probe_human_id]

            all_train_human_resemblance_counts = {}
            for train_human_d in human_label_data_train:
                train_human_id = train_human_d["D_INTERVIEW"]
                train_human_resemblance_count = 0
                for train_qid in train_qids:
                    train_human_label = train_human_d[train_qid]
                    probe_human_label = probe_human_d[train_qid]
                    if train_human_label == probe_human_label:
                        train_human_resemblance_count += 1
                all_train_human_resemblance_counts[train_human_id] = train_human_resemblance_count

            # get the top k train human resemblance counts and their corresponding human ids as two lists
            top_k_train_human_resemblance_counts = sorted(all_train_human_resemblance_counts.items(
            ), key=lambda x: x[1], reverse=True)[:top_sim_to_save]
            top_k_train_human_resemblance_human_ids = [
                train_human_id for train_human_id, _ in top_k_train_human_resemblance_counts]
            all_data_to_save.append({
                "probe_human_id": probe_human_id,
                "top_similarity_human_ids": top_k_train_human_resemblance_human_ids,
                "top_similarity_counts": [all_train_human_resemblance_counts[train_human_id] for train_human_id in top_k_train_human_resemblance_human_ids]
            })

        if is_refined:
            write_standard_data(
                all_data_to_save, f"results/baselines/resemblance_top_1_count/refined_probe_setup_{probe_setup_id}_top_{top_sim_to_save}.jsonl")
        else:
            write_standard_data(
                all_data_to_save, f"results/baselines/resemblance_top_1_count/probe_setup_{probe_setup_id}_top_{top_sim_to_save}.jsonl")


def compute_resemblance_top_k_count_baseline(is_refined=False, top_sim_to_save=500, top_k=10):
    probe_human_ids, human_label_data_train, human_label_data_test_map, statements_meta_data_map, human_label_data_map = load_data(
        is_refined)
    all_accs = []
    for probe_setup_id in range(3):
        if is_refined:
            data_to_load = f"results/baselines/resemblance_top_1_count/refined_probe_setup_{probe_setup_id}_top_{top_sim_to_save}.jsonl"
        else:
            data_to_load = f"results/baselines/resemblance_top_1_count/probe_setup_{probe_setup_id}_top_{top_sim_to_save}.jsonl"
        data = load_standard_data(data_to_load, is_print=False)

        print(f"========== Probe Setup {probe_setup_id} ==========")
        for top_k in range(1, top_k + 1):
            all_accs = []
            for sim_d in data:
                probe_human_id = sim_d["probe_human_id"]
                top_similarity_human_ids = sim_d["top_similarity_human_ids"]
                top_k_sim_human_id = top_similarity_human_ids[:top_k]

                for probe_qid in get_all_probe_qids(probe_setup_id):
                    probe_human_label = human_label_data_map[probe_human_id][probe_qid]
                    top_k_sim_human_labels = [
                        human_label_data_map[top_sim_human_id][probe_qid] for top_sim_human_id in top_k_sim_human_id]
                    top_k_sim_human_label = max(
                        set(top_k_sim_human_labels), key=top_k_sim_human_labels.count)
                    if probe_human_label == top_k_sim_human_label:
                        all_accs.append(1)
                    else:
                        all_accs.append(0)

            # print(f"Probe Setup {probe_setup_id} top_k={top_k}: {sum(all_accs) / len(all_accs)}")
            print(sum(all_accs) / len(all_accs))


if __name__ == "__main__":
    # compute_majority_vote_baseline(is_refined=True)
    # prepare_resemblance_top_1_matrices_baseline(is_refined=True, top_sim_to_save=500)
    # compute_resemblance_top_k_matrices_baseline(is_refined=True, top_sim_to_save=500, top_k=35)
    # compute_resemblance_top_k_matrices_baseline(is_refined=False, top_sim_to_save=500, top_k=35)

    # prepare_resemblance_top_1_count_baseline(is_refined=False, top_sim_to_save=100)

    compute_resemblance_top_k_count_baseline(is_refined=False, top_sim_to_save=100, top_k=35)
    compute_resemblance_top_k_count_baseline(is_refined=True, top_sim_to_save=100, top_k=35)
