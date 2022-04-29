import json
import re
from nltk.tokenize import word_tokenize
import csv


def search_entity(sentence):
    e1 = re.findall(r'<e1>(.*)</e1>', sentence)[0]
    e2 = re.findall(r'<e2>(.*)</e2>', sentence)[0]
    sentence = sentence.replace('<e1>' + e1 + '</e1>', ' <e1> ' + e1 + ' </e1> ', 1)
    sentence = sentence.replace('<e2>' + e2 + '</e2>', ' <e2> ' + e2 + ' </e2> ', 1)
    sentence = word_tokenize(sentence)
    sentence = ' '.join(sentence)
    sentence = sentence.replace('< e1 >', '<e1>')
    sentence = sentence.replace('< e2 >', '<e2>')
    sentence = sentence.replace('< /e1 >', '</e1>')
    sentence = sentence.replace('< /e2 >', '</e2>')
    sentence = sentence.split()

    assert '<e1>' in sentence
    assert '<e2>' in sentence
    assert '</e1>' in sentence
    assert '</e2>' in sentence

    return sentence


def convert(path_src, path_des):
    with open(path_src, 'r', encoding='utf-8') as fr:
        data = fr.readlines()

    print(len(data))

    with open(path_des, 'w', encoding='utf-8') as fw:
        for i in range(0, len(data), 4):
            id_s, sentence = data[i].strip().split('\t')
            sentence = sentence[1:-1]
            sentence = search_entity(sentence)
            meta = dict(
                id=id_s,
                relation=data[i + 1].strip(),
                sentence=sentence,
                comment=data[i + 2].strip()[8:]
            )
            json.dump(meta, fw, ensure_ascii=False)
            fw.write('\n')


def readJson(readPath, savePath):
    data = []
    f = open(readPath, 'r', encoding='utf-8')
    for lines in f.readlines():
        temp = json.loads(lines)
        data.append(temp)

    with open(savePath, 'w', encoding='utf-8') as fw:
        writer = csv.writer(fw)
        for i in range(len(data)):
            ID, relation, sentence, comment = data[i]["id"], data[i]["relation"], " ".join(data[i]["sentence"]), data[i]["comment"]
            writer.writerow([ID,relation,sentence,comment])

if __name__ == '__main__':
    print("begin convert json file...")

    path_train = 'train.json'
    path_test = 'test.json'

    readJson(path_train, 'train.csv')

    '''
    convert(path_train, 'train2.json')
    convert(path_test, 'test2.json')
    print("json file prepared !!!")
    '''

    # data\SemEval2010_task8_all_data\SemEval2010_task8_training\TRAIN_FILE.TXT
