import transformers

DEVICE = 'cpu'
MAX_LEN = 32
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 1
EPOCHS = 1
BERT_PATH = 'bert-base-uncased'
MODEL_PATH = '/root/docker_data/model.bin'
TRIAINING_FILE = '/root/docker_data/imdb.csv'
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH, 
    do_lowercase=True)