import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None,
                        help="Comma-separated list of GPU IDs to use, -1 means using all")
    parser.add_argument("--dataset", type=str, default="data")
    parser.add_argument("--llm", type=str, default='gpt2', help='LLM name')
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--test_epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=12, help="Random seed")
    parser.add_argument("--num_layer", type=int, default=None)
    parser.add_argument("--loc_dim", type=int, default=128)
    parser.add_argument("--user_dim", type=int, default=512)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--numf_u", type=int, default=3)
    parser.add_argument("--numf_l", type=int, default=3)

    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--lora", action='store_true')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--text", action='store_true')
    parser.add_argument("--id", action='store_true')
    parser.add_argument("--prompt", action='store_true')
    parser.add_argument("--user_text", action='store_true')
    parser.add_argument("--user_id", action='store_true')
    parser.add_argument("--profile", action='store_true')
    parser.add_argument("--aux_loss_expert", action='store_true')
    parser.add_argument("--verbose", action='store_true', help='Tqdm verbose', default=True)
    parser.add_argument("--only_text", action='store_true')
    parser.add_argument("--no_llm", action='store_true')
    parser.add_argument("--no_aux", action='store_true')
    parser.add_argument("--no_user", action='store_true')
    parser.add_argument("--load4bit", action='store_true')

    args = parser.parse_args()

    if args.llm == 'llama3':
        args.llm_id = '../LLM/Meta-Llama-3-8B'
    elif args.llm == 'llama2':
        args.llm_id = '../LLM/Llama-2-7b-hf'
    elif args.llm == 'phi2':
        args.llm_id = '../LLM/phi-2'
    elif args.llm == 'llama3.2-3':
        args.llm_id = '../LLM/Llama-3.2-3B'
    elif args.llm == 'llama3.2-1':
        args.llm_id = '../LLM/Llama-3.2-1B'
    elif args.llm == 'gpt2':
        args.llm_id = '../LLM/gpt2'
    if args.test_epoch is None:
        args.test_epoch = args.epoch
    args.dataset = 'dataset/'+args.dataset

    return args
