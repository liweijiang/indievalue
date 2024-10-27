from src.utils.main_utils import *


def main(demo_mode="polar", probe_mode="polar"):
    model_names = ["p50_p1-r50_p1-p50_r1-r50_r1-n200_full",
                   "p100_p1-r100_p1-p100_r1-r100_r1-n200_full",
                   "p150_p1-r150_p1-p150_r1-r150_r1-n200_full",
                   "p200_p1-r200_p1-p200_r1-r200_r1-n200_full",
                   "p200_p1-r200_p1-p200_r1-r200_r1-prand_p1-rrand_p1-prand_r1-rrand_r1-n100_full"]

    for model_name in model_names: 
        for num_demo in [25, 50, 75, 100, 125, 150, 175, 200]:        
            for probe_setup_id in range(3):
                raw_test_data_path = f"results/finetune_expts/{model_name}/{demo_mode}{num_demo}_{probe_mode}_test_{probe_setup_id}/raw_outputs.jsonl"
                test_model_gen_data = load_standard_data(raw_test_data_path, is_print=False)

                raw_val_data_path = f"results/finetune_expts/{model_name}/{demo_mode}{num_demo}_{probe_mode}_val_{probe_setup_id}/raw_outputs.jsonl"
                val_model_gen_data = load_standard_data(raw_val_data_path, is_print=False)

                eval_model_gen_data = test_model_gen_data + val_model_gen_data
                save_path = f"results/finetune_expts/{model_name}/{demo_mode}{num_demo}_{probe_mode}_eval_{probe_setup_id}/raw_outputs.jsonl"
                write_standard_data(eval_model_gen_data, save_path)

if __name__ == "__main__":
    main()
