import pandas as pd
# Read csv
filename = 'D:\\BMES_BERT_output.csv'

df = pd.read_csv(filename)
def cws_evaluate_word_PRF(y_pred, y, evltag=["E","S"]):
    # transfer the string to list
    y_pred = [int(i) for i in y_pred.strip('[]').split(',')]
    y = [int(i) for i in y.strip('[]').split(',')]

    # transfer "3" to“E”，"0" to“S”
    y_pred = ['E' if i == 3 else 'S' if i == 0 else 'T' for i in y_pred]
    y= ['E' if i == 3 else 'S' if i == 0 else 'T'  for i in y]

    cor_num = 0
    yp_wordnum = sum([y_pred.count(c) for c in evltag])
    yt_wordnum = sum([y.count(c) for c in evltag])
    start = 0
    for i in range(len(y)):
        if y[i] in evltag:
            flag = True
            for j in range(start, i+1):
                if y[j] != y_pred[j]:
                    flag = False
            if flag:
                cor_num += 1
            start = i+1

    P = cor_num / float(yp_wordnum) if yp_wordnum > 0 else -1
    R = cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    try:
        F = 2 * P * R / (P + R)
    except:
        F = 0
    return P, R, F



# original 
total_P = 0
total_R = 0
total_F = 0

# Store
for index, row in df.iterrows():
    P, R, F = cws_evaluate_word_PRF(row['prediction'], row['label'])
    total_P += P
    total_R += R
    total_F += F

# calculate the average
avg_P = total_P / len(df)
avg_R = total_R / len(df)
avg_F = total_F / len(df)

print(f"Average Precision: {avg_P}, Average Recall: {avg_R}, Average F1 Score: {avg_F}")
import csv
def read_csv_file(filename):
    y_pred_list = []
    y_list = []
    sentence_list = []

    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row['text']
            label = eval(row['label'])
            prediction = eval(row['prediction'])

            y_pred_list.append(prediction)
            y_list.append(label)
            sentence_list.append(list(text))

    return y_pred_list, y_list, sentence_list

# the filepath of the csv files
filename = 'D:\\BMES_BERT_output.csv'

y_pred_list, y_list, sentence_list = read_csv_file(filename)

def read_dictionary_file(filename):
    word2id = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            word = line.strip()
            word2id[word] = index
    return word2id

# path of dictionary
filename2 = 'D:\\Evahan_training_words.utf8.txt'

# generate word2id dictionary
word2id = read_dictionary_file(filename2)

def cws_evaluate_OOV(y_pred_list, y_list, sentence_list, word2id):
    cor_num = 0
    yt_wordnum = 0
    for y_pred, y, sentence in zip(y_pred_list, y_list, sentence_list):
        start = 0
        y_pred = ['E' if i == 3 else 'S' if i == 0 else 'T' for i in y_pred]
        y= ['E' if i == 3 else 'S' if i == 0 else 'T' for i in y]
        for i in range(len(y)):
            if y[i] == 'E' or y[i] == 'S':
                word = ''.join(sentence[start:i+1])
                if word in word2id or "*"  in word:
                    start = i + 1
                    continue
                flag = True
                yt_wordnum += 1
                for j in range(start, i+1):
                    if y[j] != y_pred[j]:
                        flag = False
                if flag:
                    cor_num += 1
                else:
                    #print(word)
                    None
                start = i + 1

    OOV = cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    print(cor_num)
    print(float(yt_wordnum))
    return OOV

OOV_recall = cws_evaluate_OOV(y_pred_list, y_list, sentence_list, word2id)

print(f"OOV Recall: {OOV_recall}")
