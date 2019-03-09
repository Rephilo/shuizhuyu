from numpy import *
import re


# NB = Navie Bayes


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return posting_list, class_vec


def create_vocab_list(dataset):
    vocab_set = set([])
    for document in dataset:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!", word)
    return return_vec


def trainNB0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p_0num = ones(num_words)
    p_1num = ones(num_words)
    p_0denom = 2.0
    p_1denom = 2.0
    for i in range(num_train_docs):
        if (train_category[i]) == 1:
            p_1num += train_matrix[i]
            p_1denom += sum(train_matrix[i])
        else:
            p_0num += train_matrix[i]
            p_1denom += sum(train_matrix[i])
    p_1vect = log(p_1num / p_1denom)
    p_0vect = log(p_0num / p_0denom)
    return p_0vect, p_1vect, p_abusive


def classifyNB(vec2classify, p_0vec, p_1vec, p_class1):
    p1 = sum(vec2classify + p_1vec) + log(p_class1)
    p0 = sum(vec2classify + p_0vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def testNB():
    list_o_posts, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_o_posts)
    train_mat = []
    for postin_doc in list_o_posts:
        train_mat.append(set_of_words2vec(my_vocab_list, postin_doc))
    p0v, p1v, pab = trainNB0(array(train_mat), array(list_classes))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_of_words2vec(my_vocab_list, test_entry))
    print(','.join(test_entry) + ' classified as : ' + str(classifyNB(this_doc, p0v, p1v, pab)))
    test_entry = ['stupid', 'garbage']
    this_doc = array(set_of_words2vec(my_vocab_list, test_entry))
    print(','.join(test_entry) + ' classified as : ' + str(classifyNB(this_doc, p0v, p1v, pab)))


def text_parse(big_string):
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_text():
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        word_list = text_parse(
            open('/Users/wangxiao15/Desktop/machinelearninginaction/Ch04/email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(
            open('/Users/wangxiao15/Desktop/machinelearninginaction/Ch04/email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)
    training_set = list(range(50))
    test_set = []

    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])

    train_mat = []
    train_classes = []
    for doc_index in training_set:
        train_mat.append(set_of_words2vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0v, p1v, pspam = trainNB0(array(train_mat), array(train_classes))
    error_count = 0
    for doc_index in test_set:
        word_vector = set_of_words2vec(vocab_list, doc_list[doc_index])
        if classifyNB(array(word_vector), p0v, p1v, pspam) != class_list[doc_index]:
            error_count += 1
            print("classification error", str(doc_list[doc_index]))
    print('the error rate is: ', float(error_count) / len(test_set))


if __name__ == '__main__':
    # for i in range(1, 26):
    #     try:
    #         wordList = open('/Users/wangxiao15/Desktop/machinelearninginaction/Ch04/email/ham/%d.txt' % i,
    #                         encoding='UTF-8').read()
    #     except UnicodeDecodeError:
    #         print(i)
    #     finally:
    #         print(wordList)
    spam_text()
