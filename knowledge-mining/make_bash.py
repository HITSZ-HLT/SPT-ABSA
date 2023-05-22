base_data_dir  = "/data10T/zhangyice/2022/sentiment pre-training"
base_cache_dir = "/data10T/zhangyice/.cache/huggingface/datasets"


# 使用spacy解析文本
def spacy_bash(domain, sub_dir="spacy_tokenized"):
    return f'python spacy_tokenize.py --input_dir "{base_data_dir}/raw/{domain}" --batch_size 32 --cache_dir "{base_cache_dir}/{sub_dir}/{domain}" --output_dir "{base_data_dir}/{sub_dir}/{domain}"'


# 在大量文本上构建词典，为判断单词的情感极性做准备
def vocab_bash(domain, sub_dir='vocab'):
    return f'python build_vocab_for_polarity_assign.py --input_dir "{base_data_dir}/spacy_tokenized/{domain}/" --cache_dir "{base_cache_dir}/{sub_dir}/{domain}/ --output_dir "{base_data_dir}/{sub_dir}/{domain}"'


# 为语料中的所有形容词赋予情感极性
def polarity_bash(domain, sub_dir='polarity'):
    return f'python polarity_assign.py --train_dir "{base_cache_dir}/vocab/{domain}" --output_dir "{base_data_dir}/{sub_dir}/{domain}"'


# 为500k文本构建词典及路径词典，为情感知识挖掘做准备
def vocab_path_bash(domain, sub_dir='vocab_path'):
    return f'python build_vocab_pathmanager.py --input_file_name "{base_data_dir}/spacy_tokenized/{domain}/100k_1_5.json" --cache_dir "{base_cache_dir}/{sub_dir}/{domain}/" --max_examples 500_000 --output_dir "{base_data_dir}/{sub_dir}/{domain}"'


# 运行情感知识挖掘算法
def mine_bash(domain, sub_dir='lexicon_and_rule'):
    return f'python mine.py --train_dir "{base_data_dir}/vocab_path/{domain}" --output_dir "{base_data_dir}/{sub_dir}/{domain}" --test_file_name "../asc_ae_aoe/data/14lap/test.json" --max_examples 500_000 --at_thre 0.4'


# 标注数据
def annotate_bash(domain, sub_dir='annotate'):
    return f'python annotate.py --data_dir "{base_data_dir}/spacy_tokenized/{domain}/" --cache_dir "{base_cache_dir}/{sub_dir}/{domain}/" --output_dir "{base_data_dir}/{sub_dir}/{domain}/" --vocab_path_dir "{base_data_dir}/vocab_path/{domain}" --annotator_file "{base_data_dir}/lexicon_and_rule/{domain}/9/annotator.json" --polarity_file "{base_data_dir}/polarity/{domain}/polarity.json"'



for domain in (
    'yelp',
    'amazon/Cell_Phones_and_Accessories',
    'amazon/All_Beauty',
    'amazon/AMAZON_FASHION',
    'amazon/Appliances',
    'amazon/Arts_Crafts_and_Sewing',
    'amazon/Automotive',
    'amazon/Books',
    'amazon/CDs_and_Vinyl',
    'amazon/Clothing_Shoes_and_Jewelry',
    'amazon/Digital_Music',

    'amazon/Electronics',
    'amazon/Gift_Cards',
    'amazon/Grocery_and_Gourmet_Food',
    'amazon/Home_and_Kitchen',
    'amazon/Industrial_and_Scientific',
    'amazon/Kindle_Store',
    'amazon/Luxury_Beauty',
    'amazon/Magazine_Subscriptions',
    'amazon/Movies_and_TV',
    'amazon/Musical_Instruments',
    'amazon/Office_Products',
    'amazon/Patio_Lawn_and_Garden',
    'amazon/Pet_Supplies',
    'amazon/Prime_Pantry',
    'amazon/Sports_and_Outdoors',
    'amazon/Toys_and_Games',
    'amazon/Video_Games'
):
    print(annotate_bash(domain))
    print()