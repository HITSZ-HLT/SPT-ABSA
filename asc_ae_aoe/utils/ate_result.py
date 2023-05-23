# from sklearn.metrics import f1_score, precision_score, recall_score
import os
import time
import ujson as json
from utils import append_new_line, save_json



class F1_Measure:
    def __init__(self):
        self.pred_set = set()
        self.true_set = set()

    def pred_inc(self, idx, preds):
        for pred in preds:
            self.pred_set.add((idx, tuple(pred)))
            
    def true_inc(self, idx, trues):
        for true in trues:
            self.true_set.add((idx, tuple(true)))
            
    def report(self):
        self.f1, self.p, self.r = self.cal_f1(self.pred_set, self.true_set)
        return self.f1
    
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise NotImplementedError

    def cal_f1(self, pred_set, true_set):
        intersection = pred_set.intersection(true_set)
        
        _p = len(intersection) / len(pred_set) if pred_set else 1
        _r = len(intersection) / len(true_set) if true_set else 1
        f1 = 2 * _p * _r / (_p + _r) if _p + _r else 0

        return f1, _p, _r



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
            ID = output['id']
            preds = output['preds']

            for _ID in ID:
                if _ID not in data:
                    example = examples[_ID]
                    data[_ID] = {
                        '_ID': _ID,
                        'sentence': example['sentence'],
                        'aspects':  example['aspects'],
                        'aspect_preds': set()
                    }

            for _ID, spans in zip(ID, preds):
                for _start, _end in spans:
                    data[_ID]['aspect_preds'].add((_start, _end))

        return cls(data)

    def cal_metric(self):
        # predictions  = []
        # ground_truth = []

        # for ID in self.data:
        #     example = self.data[ID]
        #     for g in example['aspects']:
        #         if len(g) == 0:
        #             continue
        #         start, end = g[:2]
        #         ground_truth.append((start, end))

        #     predictions.extend(example['aspect_preds'])

        f1 = F1_Measure()

        for ID in self.data:
            example = self.data[ID]
            g = [a[:2] for a in example['aspects']]
            p = example['aspect_preds']
            f1.true_inc(ID, g)
            f1.pred_inc(ID, p)

        f1.report()

        self.detailed_metrics = {
            'f1': f1['f1'],
            'recall': f1['r'],
            'precision': f1['p'],
        }
        self.monitor = self.detailed_metrics['f1']

    def report(self):
        for metric_names in (('precision', 'recall', 'f1'),):
            for metric_name in metric_names:
                value = self.detailed_metrics[metric_name] if metric_name in self.detailed_metrics else 0
                print(f'{metric_name}: {value:.4f}', end=' | ')
            print()

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
        self.data['info'] = {
            'time': time.strftime('%Y-%m-%d %H_%M_%S', time.localtime()),
            'model_name_or_path': model_name_or_path,
            'subname': subname,
            'dataset': dataset,
            'seed': seed,
            'metric': self.detailed_metrics
        }

    def save(self, output_dir, subname, dataset, seed):
        output_file_name = os.path.join(output_dir, f'dataset={dataset},seed={seed},m={subname}', 'result.json')
        
        info = self.data.pop('info')
        examples = [info]
        for example in self.data.values():
            example['aspect_preds'] = sorted(example['aspect_preds'])
            examples.append(example)

        save_json(examples, output_file_name)
