#!/usr/bin/env python

from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import *
from read_database import *
from check_similarity import *
import numpy as np
from intent_classifier import *
from laserembeddings import Laser
import warnings
warnings.filterwarnings("ignore")
import random
from sklearn.metrics import *
from APIOptimizer import *

##########################################################################
# LOAD CONFIGN
with open("config\chatbot_config.txt", "r",encoding='utf-8') as file:
    content_config = file.read().split("\n")

for line in content_config:
    if line.startswith("NUMBER_OF_RECOMMENDATION_RESULT"):
        NUMBER_OF_RECOMMENDATION_RESULT = float(line.split("=")[1])
    if line.startswith("THRESHOLD_INTENT_CONFIDENCE_DEFAULT"):
        THRESHOLD_INTENT_CONFIDENCE_DEFAULT = float(line.split("=")[1])
    if line.startswith("THRESHOLD_INTENT_CONFIDENCE_TRACKING"):
        THRESHOLD_INTENT_CONFIDENCE_TRACKING = float(line.split("=")[1])
    if line.startswith("THRESHOLD_INTENT_CONFIDENCE_GREETING"):
        THRESHOLD_INTENT_CONFIDENCE_GREETING = float(line.split("=")[1])
    if line.startswith("THRESHOLD_ERROR_FOR_AGENT"):
        THRESHOLD_ERROR_FOR_AGENT = float(line.split("=")[1])
    if line.startswith("THRESHOLD_RETURN_CONFIDENCE"):
        THRESHOLD_RETURN_CONFIDENCE = float(line.split("=")[1])
    if line.startswith("THRESHOLD_INTENT_FOR_NOT_TRACKING"):
        THRESHOLD_INTENT_FOR_NOT_TRACKING = float(line.split("=")[1])


print("NUMBER_OF_RECOMMENDATION_RESULT",NUMBER_OF_RECOMMENDATION_RESULT)
print("THRESHOLD_INTENT_CONFIDENCE_DEFAULT",THRESHOLD_INTENT_CONFIDENCE_DEFAULT)
print("THRESHOLD_INTENT_CONFIDENCE_TRACKING",THRESHOLD_INTENT_CONFIDENCE_TRACKING)
print("THRESHOLD_INTENT_CONFIDENCE_GREETING",THRESHOLD_INTENT_CONFIDENCE_GREETING)
print("THRESHOLD_ERROR_FOR_AGENT",THRESHOLD_ERROR_FOR_AGENT)
print("THRESHOLD_RETURN_CONFIDENCE",THRESHOLD_RETURN_CONFIDENCE)
print("THRESHOLD_INTENT_FOR_NOT_TRACKING",THRESHOLD_INTENT_FOR_NOT_TRACKING)

laser_model = Laser()


# CAC GIA TRI CAN KHOI TAO LAI KHI REFRESH
intent_model = read_trained_model()
vectorizer = read_vectorize()
list_label = read_list_label()
max_length = read_max_length()
answer_database, answer_template = read_answer_database()
private_knowledge_df = read_knowledge(path_private="Data/Data.xlsx",
                                      path_share="Data/Data.xlsx")
synonyms_dictionary = read_share_knowledge(path_share="Data/Data.xlsx")

# private_knowledge_df = read_knowledge(path_private="Data/Data_VH.xlsx",
#                                       path_share="Data/Data_VH.xlsx")
# synonyms_dictionary = read_share_knowledge(path_share="Data/Data_VH.xlsx")

Out_intent_model = read_trained_model(path="pickle_folder/Out_intent_model.h5")
Out_vectorizer = read_vectorize(path="pickle_folder/Out_vectorizer.pickle")
Out_max_length = read_max_length(path="pickle_folder/Out_max_length.pickle")
### add new
question_out_domain, list_ans_out_domain = read_question_out_domain_have_ans(synonyms_dictionary)
question_out_domain_matrix = laser_model.embed_sentences(question_out_domain, lang="vi")


dict_description,dict_description_preprocessing = read_dict_intent_description(synonyms_dictionary)
dict_description_matrix = dict()
for intent in dict_description:
    print("="*20)
    print(intent)
    print(len(dict_description[intent]))
    dict_description_matrix[intent] = laser_model.embed_sentences(dict_description[intent], lang="vi")


description,description_clean,description_clean2 = read_description(answer_database,answer_template,synonyms_dictionary)
description_matrix =  laser_model.embed_sentences(description_clean2, lang='vi')
list_product = read_product(answer_database,answer_template)

#THEM
list_product = read_product(answer_database,answer_template)
list_product_clean = [processing(product_name,synonyms_dictionary) for product_name in list_product]
#THEM
product_name_matrix = laser_model.embed_sentences(list_product_clean, lang='vi')

# Chat xam CSDL

answer_database_chatxam = read_answer_chatxam()
question_chat_xam, template_chatxam = read_question_chatxam(synonyms_dictionary,file_name="Data/Data_Funny.xlsx")
question_chat_xam_matrix = laser_model.embed_sentences(question_chat_xam, lang="vi")

accent_restore = read_accent_model()

def cacluate_accuracy_subset(intent_template, type=1):

    data = pd.read_excel("Data/Data.xlsx", sheet_name="Question", skiprows=2)
    # data = pd.read_excel("Data/Data_VH.xlsx", sheet_name="Question", skiprows=2)
    X_data = data.loc[data['Pattern Template'] == intent_template]["Question"].tolist()

    Answer_database = [get_answer_based_template(answer_database,intent_template)] * len(X_data)
    query_list = []
    response_list = []
    answer_list = []
    preprocessing_list = []
    count = 0
    score_list = []
    confidence_score = []
    session_id = random.randint(0, 100000000000000)
   # #print("Start running prediction ....")
    for index, user_query in enumerate(X_data):
        intent1,confidence_score1,response = get_response_api(user_query, session_id,
                     intent_model, vectorizer, list_label, max_length,
                     Out_intent_model, Out_vectorizer, Out_max_length,
                     answer_database, answer_template,
                     description_matrix,description_clean,description,
                     list_product, list_product_clean, product_name_matrix,
                     question_chat_xam, template_chatxam, question_chat_xam_matrix, answer_database_chatxam,
                     laser_model, private_knowledge_df, synonyms_dictionary,accent_restore,
                     question_out_domain_matrix, list_ans_out_domain,
                    dict_description,dict_description_preprocessing,
                     dict_description_matrix,
                     THRESHOLD_INTENT_CONFIDENCE_DEFAULT,
                     THRESHOLD_INTENT_CONFIDENCE_TRACKING,
                     THRESHOLD_INTENT_CONFIDENCE_GREETING,
                     THRESHOLD_ERROR_FOR_AGENT,
                     NUMBER_OF_RECOMMENDATION_RESULT,
                     THRESHOLD_RETURN_CONFIDENCE,
                     THRESHOLD_INTENT_FOR_NOT_TRACKING)
        if Answer_database[index] in response:
            count += 1
        else:
            ##print("user_query", user_query)
            ##print()
            query_list.append(user_query)
            response_list.append(response)
            answer_list.append(Answer_database[index])
            score_list.append(intent1)
            confidence_score.append(confidence_score1)
    if int(round(count / len(X_data) * 100, 4)) != 100:
        print(intent_template)
        try:
            print("Số lượng mẫu trả lời đúng: ", count)
            print("Accuracy: ", round(count / len(X_data) * 100, 4))
        except:
            print("Số lượng mẫu trả lời đúng: ", count)
            #rint("Accuracy: 0")

        print("DANH SÁCH CÁC CÂU LỖI")
        print("------------------------------------------------------------------------")
        for index, item in enumerate(query_list):
            print(item)
            print(processing(item,synonyms_dictionary))
            print("intent:", score_list[index])
            print("confidence score: ", confidence_score[index])
            #print("Response: ", response_list[index])
            print("\n")
            # #print("Response: ", response_list[index])

    return round(count / len(X_data) * 100, 4)


def verify_performace():
    data = pd.read_excel("Data/Data.xlsx", sheet_name="Question", skiprows=2)
    # data = pd.read_excel("Data/Data_VH.xlsx", sheet_name="Question", skiprows=2)
    templates = data['Pattern Template'].unique().tolist()
    #print(len(templates))
    accuracy = []
    text_accuracy_100 = ""
    count = 0
    for template in templates:
        acc = cacluate_accuracy_subset(template)
        accuracy.append(acc)
        if int(acc) == 100:
            text_accuracy_100 += template + "\n"
            count += 1
        ##print("=" * 100)

    print("Average Performance: ", sum(accuracy) / len(accuracy))
    print("số lượng intent 100% accuracy: ", count)

if __name__ == '__main__':
    intent_template = 'hỏi_đáp_nghề_nghiệp|kinh_doanh'
    cacluate_accuracy_subset(intent_template, 2)
    
    # verify_performace_base_intent(intent)

    # verify_performace()
