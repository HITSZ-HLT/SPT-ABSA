import os
import ujson as json



def _mkdir_if_not_exist(dir_name):    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def mkdir_if_not_exist(path):
    dir_name, file_name = os.path.split(path)
    if dir_name:
        _mkdir_if_not_exist(dir_name)



def yield_data_file(data_dir):
    for file_name in os.listdir(data_dir):
        yield file_name, os.path.join(data_dir, file_name)



def save_json(json_obj, file_name):
    mkdir_if_not_exist(file_name)
    with open(file_name, mode='w', encoding='utf-8-sig') as f:
        json.dump(json_obj, f, indent=4)



def load_json(file_name):
    with open(file_name, mode='r', encoding='utf-8-sig') as f:
        return json.load(f)



def load_line_json(file_name, N):
    count = 0

    with open(file_name, mode='r') as f:
        for line in f:
            try:
                yield json.loads(line)
                count += 1
                if count >= N:
                    break

            except:
                # 奇怪的bug
                a, b = line.split('}{"ID":')

                yield json.loads(a+'}')
                count += 1
                if count >= N:
                    break

                yield json.loads('{"ID":'+b)
                count += 1
                if count >= N:
                    break



def load_mpqa(file_name, polaritys=('neutral', 'negative', 'positive')):
    with open(file_name, 'r+') as f:
        for line in f:
            line = line.strip().split()
            word = line[2].split('=')[1]
            pos  = line[3].split('=')[1]
            priorpolarity = line[-1].split('=')[1]

            if priorpolarity in polaritys:
                if pos == 'anypos':
                    for pos in ('noun', 'verb', 'adj', 'adverb'):
                        yield word, pos, priorpolarity
                else:
                    yield word, pos, priorpolarity



def load_mpqa2(file_name, polaritys=('neutral', 'negative', 'positive')):
    with open(file_name, 'r+') as f:
        for line in f:
            line = line.strip().split()
            intensity = line[0].split('=')[1]
            word = line[2].split('=')[1]
            pos  = line[3].split('=')[1]
            priorpolarity = line[-1].split('=')[1]

            if priorpolarity in polaritys:
                if pos == 'anypos':
                    for pos in ('noun', 'verb', 'adj', 'adverb'):
                        yield word, pos, intensity, priorpolarity
                else:
                    yield word, pos, intensity, priorpolarity