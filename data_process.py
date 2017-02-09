import numpy as np
import random

def create_dictionary():
    dictionary = []
    with open('words.txt', 'r') as file:
        whole = file.readlines()
        for line in whole[:45000]:
            dictionary.append(line[:-1])
    print("end of words dictionary")
    return dictionary

def data_handling(filename):
    whole_file = []
    dictionary = create_dictionary()
    print("Start of file", filename)
    with open(filename, 'r', encoding="utf-8-sig") as input_file:
        whole = input_file.readlines()
        max_length = 120
        for line in whole[:2000]:
            line = line.rstrip()
            words = line.split(" ")
            sentence = []
            for word in words[:-1]:
                if word[-1:] == ("," or "." or "!" or "?" or ":" or ";"):
                    word = word[:-1]
                word = word.lower()
                if word in dictionary:
                    properform = np.zeros(len(dictionary))
                    np.put(properform, dictionary.index(word), 1)
                    sentence.append(properform)
            while (len(sentence) < max_length):
                properform = np.zeros(len(dictionary))
                sentence.append(properform)
            if words[-1:] == ['0']:
                sentence_class = [sentence, np.repeat(np.zeros(1), 120)]
            if words[-1:] == ['1']:
                sentence_class = [sentence, np.repeat(np.ones(1), 120)]
            whole_file.append(sentence_class)
    print("End of file")
    return whole_file

def tweet_splitter():
    test_size = 0.1
    whole_test = []
    whole_test += data_handling('good.txt')
    whole_test += data_handling('Hate.txt')
    random.shuffle(whole_test)
    whole_test = np.array(whole_test)
    whole_test_size = int(test_size*len(whole_test))
    train_input = list(whole_test[:,0][:-whole_test_size])
    train_output = list(whole_test[:,1][:-whole_test_size])
    validate_input = list(whole_test[:,0][-whole_test_size:])
    validate_output = list(whole_test[:,1][-whole_test_size:])

    return train_input, train_output, validate_input, validate_output

