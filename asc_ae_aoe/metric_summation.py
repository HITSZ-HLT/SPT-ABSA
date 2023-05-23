from utils import load_line_json


def main(args):
    metrics = load_line_json(args.source_file)
    metric_dict = {}
    for metric in metrics:
        if metric['subname'] == args.subname:
            dataset = metric['dataset']
            seed    = metric['seed']
            metric_dict[(dataset, seed)] = metric['metric']
            model_name_or_path = metric['model_name_or_path']

    f1_list = []

    datasets = ('14res', '14lap')
    
    if args.mode == '40_80':
        seeds = (40, 50, 60, 70, 80)
    elif args.mode == '140_180':
        seeds = (140, 150, 160, 170, 180)    
    elif args.mode == '40_80_140_180':
        seeds = (40, 50, 60, 70, 80) + (140, 150, 160, 170, 180)    
    else:
        raise NotImplementedError()

    # print(metric_dict.keys())
    for dataset in datasets:
        f1 = [metric_dict[(dataset, seed)]['f1']*10000 for seed in seeds]
        print(dataset, f1)
        f1 = int(sum(f1) / len(seeds))
        f1_list.append(f1)
    
    # model_name_or_path = metric_dict[(datasets[0], seeds[0])]
    f1_list.append(sum(f1_list)/len(datasets))
    print()
    print(model_name_or_path)
    for _f1 in f1_list:
        print(_f1, end='    ')
    print()




if __name__ == '__main__':
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type=str)
    parser.add_argument('--subname', type=str)
    parser.add_argument('--mode', type=str, default='40_80_140_180')

    args = parser.parse_args()

    if len(args.source_file) == 3:
        args.source_file = f"/data/zhangyice/2022/sentiment pre-training/downstream-tasks/{args.source_file}/output/performance.txt"

    main(args)



# python metric_summation.py --source_file "/data10T/zhangyice/2022/sentiment pre-training/downstream-tasks/asc/output/performance.txt" --subname test