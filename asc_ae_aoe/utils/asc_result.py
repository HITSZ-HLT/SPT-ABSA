import os
import time
import ujson as json
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import append_new_line 
from utils.asc_pair_datamodule import polarity_map

# polarity_map = {
#     'NEG': 0,
#     'NEU': 1,
#     'POS': 2,
#     0: 'NEG',
#     1: 'NEU',
#     2: 'POS',
# }

class Result:
    def __init__(self, data):
        self.data = data

    def __ge__(self, other):
        return self.monitor >= other.monitor

    def __gt__(self, other):
        return self.monitor >  other.monitor

    @classmethod
    def parse_from(cls, outputs, examples):
        data = {}
        examples = {example['ID']: example for example in examples}

        for output in outputs:
            ID, start, end = output['id_start_end']
            preds = output['preds']

            for _ID in ID:
                if _ID not in data:
                    example = examples[_ID]
                    data[_ID] = {
                        '_ID': _ID,
                        'sentence': example['sentence'],
                        'aspects': example['aspects'],
                        'aspect_preds': set()
                    }

            for _ID, _start, _end, _polarity in zip(ID, start, end, preds):
                data[_ID]['aspect_preds'].add((_start, _end, polarity_map[_polarity]))

        return cls(data)

    def cal_metric(self):
        predictions  = []
        ground_truth = []

        for ID in self.data:
            example = self.data[ID]
            g = sorted([a for a in example['aspects']])
            p = sorted(example['aspect_preds'])
            assert len(g) == len(p), f'{len(g)} != {len(p)}, {g}, {p}, {example}'

            # print(g)
            # print(p)

            for _g, _p in zip(g, p):
                s1, e1, p1 = _g[:3]
                s2, e2, p2 = _p[:3]
                assert s1 == s2 and e1 == e2
                ground_truth.append(p1)
                predictions.append(p2)

        print('--------------------')
        # print(len(ground_truth))
        # print(predictions)

        self.detailed_metrics = {
            'f1': f1_score(ground_truth, predictions, average='macro'),
            'recall': recall_score(ground_truth, predictions, average='macro'),
            'precision': precision_score(ground_truth, predictions, average='macro'),
        }
        self.monitor = self.detailed_metrics['f1']

    def save_metric(self, output_dir, model_name_or_path, subname, dataset, seed):
        performance_file_name = os.path.join(output_dir, 'performance.txt')
        print('save performace to', performance_file_name)
        append_new_line(performance_file_name, json.dumps({
            'time': time.strftime('%Y-%m-%d %H_%M_%S', time.localtime()),
            'model_name_or_path': model_name_or_path,
            'subname': subname,
            'dataset': dataset,
            'seed': seed,
            'metric': self.detailed_metrics
        }))

    def report(self):
        for metric_names in (('precision', 'recall', 'f1'),):
            for metric_name in metric_names:
                value = self.detailed_metrics[metric_name] if metric_name in self.detailed_metrics else 0
                print(f'{metric_name}: {value:.4f}', end=' | ')
            print()


