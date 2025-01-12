import pandas as pd
import ast
import re
# CSV filepath
filename = "D:\\WBD_BERT_output.csv"
# dictionary filepath
filename2 = 'D:\\Evahan_training_words.utf8.txt'
# This code will transfer the labels in to segmented texts, and then label the segemented texts into BMES
# Then, compare the results in the same level.
df = pd.read_csv(filename)
predict_processed_text = []
real_processed_text = []
for index, row in df.iterrows():
        predict_text = list(row['text'])
        real_text = list(row['text'])
        predict_label = ast.literal_eval(row['prediction'])  # transfer strings to list
        real_label = ast.literal_eval(row['label'])
        if len(predict_text) > len(predict_label):
            print(predict_text)
        for i in range(len(predict_label)):
            if predict_label[i] == 1:
                try:
                    predict_text[i] = predict_text[i] + ' '
                except:
                    break
        predict_processed_text.append(''.join(predict_text))
        if len(real_text) > len(real_label):
            print(real_text)
        for i in range(len(real_label)):
            if real_label[i] == 1:
                try:
                    real_text[i] = real_text[i] + ' '
                except:
                    break
        real_processed_text.append(''.join(real_text))
    # write the processed results into the files 
file_predict = "predict_segmentation_result.txt"
file_real ="real_segmentation_result.txt"
with open(file_predict, 'w', encoding='utf-8') as file:
        for text in predict_processed_text:
            file.write(text + '\n')  
with open(file_real, 'w', encoding='utf-8') as file:
        for text in real_processed_text:
            file.write(text + '\n')   
import csv
def process_text(input_file1,input_file2, output_file):
        with open(input_file1, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            processed_lines = []
            digit_lines = []
            for line in lines:
                    line = line.strip()
                    processed_line = ''
                    digit_line = []
                    word_list = re.split('\\s+', line)
                    for word in word_list:
                        if len(word) ==0:
                            continue
                        if len(word) == 1:
                            processed_line+=word
                            digit_line.append(0)#S
                        elif len(word) == 2:
                            processed_line+=word[0]
                            digit_line.append(1)#B
                            processed_line+=word[1]
                            digit_line.append(3)#E
                        else:
                            processed_line+=word[0]
                            digit_line.append(1)
                            for i in range(1, len(word) - 1):
                                processed_line+=word[i]
                                digit_line.append(2)#M
                            processed_line+=word[-1]
                            digit_line.append(3)
                    digit_lines.append(digit_line)
                    processed_lines.append(line.replace(" ", ''))
        with open(input_file2, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            real_digit_lines = []
            for line in lines:
                    line = line.strip()
                    real_digit_line = []
                    word_list = re.split('\\s+', line)
                    for word in word_list:
                        if len(word) ==0:
                            continue
                        if len(word) == 1:
                            real_digit_line.append(0)#S
                        elif len(word) == 2:
                            real_digit_line.append(1)#B
                            real_digit_line.append(3)#E
                        else:
                            real_digit_line.append(1)
                            for i in range(1, len(word) - 1):
                                real_digit_line.append(2)#M
                            real_digit_line.append(3)
                    real_digit_lines.append(real_digit_line)
       
        with open(output_file, 'w', encoding='utf-8', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['text', 'prediction','label'])
                    for i in range(len(lines)):
                        writer.writerow([processed_lines[i],digit_lines[i],real_digit_lines[i]])
chuli = "chuli.csv"
process_text(file_predict,file_real,chuli)
 #We have finished the processing. Now, run evaluation process for BMES.
def cws_evaluate_word_PRF(y_pred, y, evltag=["E","S"]):
        # transfer strings to lists
        y_pred = [int(i) for i in y_pred.strip('[]').split(',')]
        y = [int(i) for i in y.strip('[]').split(',')]

        # transfer "0" for“S”，3 for“E”
        y_pred = ['E' if i == 3 else 'S' if i == 0 else 'T' for i in y_pred]
        y = ['E' if i == 3 else 'S' if i == 0 else 'T' for i in y]

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

# read csv
df = pd.read_csv(chuli)

# original
total_P = 0
total_R = 0
total_F = 0

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
import pandas as pd
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



# get data
y_pred_list, y_list, sentence_list = read_csv_file(chuli)

def read_dictionary_file(filename):
        word2id = {}
        with open(filename, 'r', encoding='utf-8') as file:
            for index, line in enumerate(file):
                word = line.strip()
                word2id[word] = index
        return word2id



# generate word2id dictionary
word2id = read_dictionary_file(filename2)

def cws_evaluate_OOV(y_pred_list, y_list, sentence_list, word2id):
        cor_num = 0
        yt_wordnum = 0
        OOV_true_word = []
        for y_pred, y, sentence in zip(y_pred_list, y_list, sentence_list):
            start = 0
            y_pred = ['E' if i == 3 else 'S' if i == 0 else 'T' for i in y_pred]
            y = ['E' if i == 3 else 'S' if i == 0 else 'T' for i in y]
            for i in range(len(y)):
                if y[i] == 'E'or y[i] == 'S':
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
                        OOV_true_word.append(word)
                        cor_num += 1
                    else:
                        #print(word)
                        None
                    start = i + 1

        OOV = cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
        print(cor_num)
        print(float(yt_wordnum))
        return OOV, OOV_true_word
# read csv
df = pd.read_csv(filename)
OOV_recall,  OOV_true_word_list = cws_evaluate_OOV(y_pred_list, y_list, sentence_list, word2id)
print(f"OOV Recall: {OOV_recall}")
print("----------------------------------------------")
