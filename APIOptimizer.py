#!/usr/bin/env python

import warnings
from intent_classifier import *
from tkinter import *
from preprocessing import *
import random
from Out_intent_classifier import  *
from langdetect import detect
from check_similarity import *
from Database import *
from add_accents import *
warnings.filterwarnings("ignore")
import time

#BO SUNG DOT 2
def check_available_in_list(check_matrix, value,laser_model):
    start4444 = time.time()
    threshold_similarity_product = 0.85
    vector_input1 = laser_model.embed_sentences(value, lang='vi')

    #TOI UU: SORT THEO MAX RỒI LẤY INDEX GIÁ TRỊ MAX.
    cosin_similarities = cos_sim_2d(vector_input1, check_matrix)
    max_value = max(cosin_similarities[0])

    end4444 = time.time()
    #print("DEBUG : Time of check_available_in_list of list: ", end4444 - start4444)
    #print(max_value)
    if max_value > threshold_similarity_product:
        return True
    else:
        return False


def check_available_in_list_chatxam(question_chatxam_matrix, value,laser_model):
    threshold_similarity_product = 0.85
    vector_input1 = laser_model.embed_sentences(value, lang='vi')

    cosin_similarities = cos_sim_2d(vector_input1, question_chatxam_matrix)
    max_value = max(cosin_similarities[0])
    max_index = list(cosin_similarities[0]).index(max_value)
    if max_value > threshold_similarity_product:
        return max_index
    else:
        return -1



# Get entity from
def get_entity_list(template_list, start,end):
    list_of_entities = []
    for item in template_list:
        entities = item.split("|")[start:end]
        if entities not in list_of_entities:
            list_of_entities.append(entities)
    return list_of_entities

# Get max value in its index from a vector numpy
def get_max2index_value(prob_array):
    max_value = max(np.array(prob_array))
    max_index = list(prob_array).index(max_value)
    return max_value, max_index

# Fill the slot
def fill_slot(intent, template, user_query ,user_query_clean, private_knowledge_df):

    fill_out_results = ""
    count_true = 0
    count_entity = 0
    for entity in template.split("|")[1:]:
        count_entity += 1
        entity = clean_text(entity)
        user_query_clean = clean_text(user_query_clean)
        #user_query = clean_text(user_query)

        if check_synonym(entity, intent, user_query_clean, private_knowledge_df) == True:
            fill_out_results += "True|"
            count_true += 1
        else:
            fill_out_results += "False|"
    fill_out_results = fill_out_results + str(count_true) + "|" + str(count_entity)

    return fill_out_results

# Fill all candidate template
def fill_candidate_templates(intent,list_template, user_query ,user_query_clean,private_knowledge_df):

    list_output = []
    for template in list_template:
        output = fill_slot(intent, template, user_query ,user_query_clean,private_knowledge_df)
        list_output.append(output)

    return list_output

# Select prority answer
def priority_answer(mask_list):
    res = []
    mask_list_inv = [list(reversed(l)) for l in mask_list]
    current_prob = [0] * len(mask_list)
    max_prob = 0

    for index_mask, mask in enumerate(mask_list_inv):
        for index, x in enumerate(mask):
            if x == False:
                current_prob[index_mask] = (index + 1) / len(mask)
                break
        max_prob = max(max_prob, current_prob[index_mask])

    for index in range(len(current_prob)):
        if current_prob[index] == max_prob:
            res.append(index)
    return res

# Get response for 1 candidate template
def get_one_response(template_promise_intent,answer_database):
    return  answer_database[template_promise_intent[0]]["Answer"]


def get_index_of_top_value(list1, topk):
    list1 = list(np.array(list1))
    count = 0
    output = []
    while (count < topk):
        x = max(list1)
        m = [i for i in range(len(list1)) if list1[i] == x]
        output += m
        count = len(output)
        list1 = [0 if v==x else v for v in list1]
    return output

def ranking_again(user_query, suggestions, suggestions_processing,topk,cosin_similarity):
    word_list = user_query.split(" ")
    list_count = []
    for index,suggest in enumerate(suggestions):
        count = 0
        for word in word_list:
            if word in suggestions_processing[index]:
                count +=1
        list_count.append(count)

    a = get_index_of_top_value(list_count, topk)
    suggestion_ranking = []
    recommendation_score = 0.0
    for i,index in enumerate(a):
        if i < topk:
            if suggestions[index] not in suggestion_ranking:
                suggestion_ranking.append(suggestions[index])
                if (list_count[index]/len(word_list) + 2*cosin_similarity[index])/3 >= 0.7:
                    recommendation_score += 1.0 
                else:
                    recommendation_score += (list_count[index]/len(word_list) + 2*cosin_similarity[index])/3
            else:
                topk +=1
        else:
            break
    recommendation_score = recommendation_score/topk
    return suggestion_ranking, recommendation_score

#BOSUNG CODE
def get_recommendation(user_query,intent,dict_description,dict_description_processing,dict_description_matrix, laser_model,answer_template, synonyms_dictionary, topk=5):
    start = time.time()
    suggestions = []
    suggestions_processing = []

    # =====================================
    if "_" in user_query:
        user_query_v = user_query
    else:
        user_query_v = processing(user_query,synonyms_dictionary)
    # =====================================

    vector_query = laser_model.embed_sentences(user_query_v, lang='vi')
    print(vector_query)
    print(vector_query.shape)
    print("co di qua hay khong")
    cosin_similarity = cos_sim_2d(vector_query, dict_description_matrix[intent])
    a = get_index_of_top_value(cosin_similarity[0], 10 )
    for index in a:
        if len(str(dict_description[intent][index]).split(" ")) > 2:
            suggestions.append(dict_description[intent][index])
            suggestions_processing.append(dict_description_processing[intent][index])
    suggestions,recommendation_score = ranking_again(user_query_v, suggestions, suggestions_processing ,topk, cosin_similarity[0])

    end = time.time()
    #print("Time for Recommendation Function: ", end - start)

    return suggestions,recommendation_score

def get_answer_based_template(answer_database,template):
    list_answer = answer_database[template]["Answer"]
    answer = random.choice(list_answer)
    return answer

def get_description_based_template(answer_database,template):
    des = answer_database[template]["Description"]
    return des

# Get response many candidate templates  theo dõi gợi ý ...
def get_template_response(state_conversation,fill_out_results, template_candidate ,answer_database, tracking=False, recomendation_threshold=5):
    # print("index at get template response")
    confidence_slot_filling_list = []

    response_list = []
    #print("state_conversation.step: ", state_conversation.step)

    # Trường hợp nhãn này chỉ có 1 template cho 1 intent
    if len(template_candidate) == 1:
        if "True" in fill_out_results[0]:
            response_list.append(template_candidate[0])
            a = fill_out_results[0].split("|")
            confidence_slot_filling = int(a[-2])/int(a[-1])
            confidence_slot_filling_list.append(confidence_slot_filling)
            # print("confidence_slot_filling_list: ", confidence_slot_filling_list)
            return False, response_list, confidence_slot_filling_list
        else:
            False, [], []
    # Có nhiều hơn templates
    else:
        # print("có nhiều hơn 1 template")
        candidate_index = []
        for index, item in enumerate(fill_out_results):
            if "False" not in item:
                candidate_index.append(index)
        if len(candidate_index) == 1:
            response_list.append(template_candidate[candidate_index[0]])
            a = fill_out_results[candidate_index[0]].split("|")
            confidence_slot_filling = int(a[-2]) / int(a[-1])
            confidence_slot_filling_list.append(confidence_slot_filling)
            # print("confidence_slot_filling_list: ", confidence_slot_filling_list)
            # print(response_list)
            return False, response_list, confidence_slot_filling_list
        elif len(candidate_index) >= 2:
            # Check the answer whether is similarity?
            list_answer = [get_answer_based_template(answer_database, template_candidate[index]) for index in candidate_index]
            if len(set(list_answer)) == 1:
                response_list.append(template_candidate[candidate_index[0]])
                a = fill_out_results[candidate_index[0]].split("|")
                confidence_slot_filling = int(a[-2]) / int(a[-1])
                confidence_slot_filling_list.append(confidence_slot_filling)
                # print("confidence_slot_filling_list: ", confidence_slot_filling_list)
                # print(response_list)
                return False, response_list, confidence_slot_filling_list
            else:
                # Kiểm tra xem là các template có full True hết. Template nào có nhiều true hơn thì sẽ được chọn đưa ra câu trả lời
                count_entity_list = [int(fill_out_results[index].split("|")[-2]) for index in candidate_index]
                max_index = count_entity_list.index(max(count_entity_list))
                max_value = count_entity_list[max_index]
                max_list = [i for i, j in enumerate(count_entity_list) if j == max_value]
                if len(max_list) < 2:
                    response_list.append(template_candidate[candidate_index[max_index]])
                    a = fill_out_results[candidate_index[max_index]].split("|")
                    confidence_slot_filling = int(a[-2]) / int(a[-1])
                    confidence_slot_filling_list.append(confidence_slot_filling)
                    # print("confidence_slot_filling_list: ", confidence_slot_filling_list)
                    # print(response_list)
                    return False, response_list, confidence_slot_filling_list
                elif len(max_list) == 2:
                    for index,iz in enumerate(max_list[:2]):
                        response_list.append(template_candidate[candidate_index[iz]])
                        a = fill_out_results[candidate_index[max_index]].split("|")
                        confidence_slot_filling = int(a[-2]) / int(a[-1])
                        confidence_slot_filling_list.append(confidence_slot_filling)
                    # print("confidence_slot_filling_list: ", confidence_slot_filling_list)
                    # print(response_list)
                    return False, response_list, confidence_slot_filling_list
                else:
                    for index,iz in enumerate(max_list[0:recomendation_threshold]):
                        response_list.append(template_candidate[candidate_index[iz]])
                        a = fill_out_results[candidate_index[iz]].split("|")
                        confidence_slot_filling = int(a[-2]) / int(a[-1])
                        confidence_slot_filling_list.append(confidence_slot_filling)
                        #print(fill_out_results[candidate_index[iz]])

                    # print("confidence_slot_filling_list: ", confidence_slot_filling_list)
                    # print(response_list)
                    return False, response_list, confidence_slot_filling_list
        else:
            # Nếu có đoạn hội thoại phía trước thì sử dụng tracking để kiểm tra xong fill lại được gì không?
            if state_conversation.step_conversation > 1 and tracking == False:
                #print("Conversation tracking")
                return True, [], []

            else:
                count_True_list = [int(item.split("|")[-2]) for item in fill_out_results]
                if any(y > 0 for y in count_True_list) == False:
                    return False, [], []

                else:
                    max_value = max(count_True_list)
                    max_value_list = []
                    for index, value in enumerate(count_True_list):
                        if value == max_value:
                            max_value_list.append(index)

                    if len(max_value_list) > 1:
                        list_priorty = priority_answer([fill_out_results[i] for i in max_value_list])
                        if len(list_priorty) == 1:
                            response_template = template_candidate[list_priorty[0]]
                            response_list.append(response_template)
                            a = fill_out_results[list_priorty[0]].split("|")
                            confidence_slot_filling = int(a[-2]) / int(a[-1])
                            confidence_slot_filling_list.append(confidence_slot_filling)
                            #print("confidence_slot_filling_list: ", confidence_slot_filling_list)
                            return False, response_list, confidence_slot_filling_list
                        else:
                            for index, iz in enumerate(list_priorty[0:recomendation_threshold]):
                                response_template = template_candidate[max_value_list[iz]]
                                response_list.append(response_template)

                                a = fill_out_results[max_value_list[iz]].split("|")
                                confidence_slot_filling = int(a[-2]) / int(a[-1])
                                confidence_slot_filling_list.append(confidence_slot_filling)
                            #print("confidence_slot_filling_list: ", confidence_slot_filling_list)
                            return False, response_list, confidence_slot_filling_list

                    elif len(max_value_list) == 1 and int(fill_out_results[max_value_list[0]][-1]) > 2:
                        response_template = template_candidate[max_value_list[0]]
                        response_list.append(response_template)

                        a = fill_out_results[max_value_list[0]].split("|")
                        confidence_slot_filling = int(a[-2]) / int(a[-1])
                        confidence_slot_filling_list.append(confidence_slot_filling)
                        #print("confidence_slot_filling_list: ", confidence_slot_filling_list)
                        return False, response_list, confidence_slot_filling_list
    return False, [], []

def cos_sim_2d(x, y):
    norm_x = x / np.linalg.norm(x, axis=1, keepdims=True)
    norm_y = y / np.linalg.norm(y, axis=1, keepdims=True)
    return np.matmul(norm_x, norm_y.T)


# EDIT CODE
# Check similarity between user query with product
def check_similarity_product_query(user_query_clean, list_product_clean,product_name_matrix,laser_model):
    start323 = time.time()
    for product_name in list_product_clean:
        if product_name in user_query_clean:
            end3232 = time.time()
            #print("Time of check_similarity_product_query: ", end3232 - start323)
            return True

    z = re.findall(r'\b' +r"(nữ|con_gái).*(nghề|ngành|)" +  r'\b', user_query_clean)
    if len(z)>0:
        return True
    threshold_similarity_product = 0.85
    vector_input1 = laser_model.embed_sentences(user_query_clean, lang='vi')

    #TOI UU: SORT THEO MAX RỒI LẤY INDEX GIÁ TRỊ MAX.
    cosin_similarities = cos_sim_2d(vector_input1, product_name_matrix)
    max_value = max(cosin_similarities[0])
    print("max value")
    print(max_value)
    end3232 = time.time()
    #print("Time of check_similarity_product_query: ", end3232 - start323)
    if max_value > threshold_similarity_product:
        return True
    else:
        return False
#gợi ý cuộc trò chuyện
def bot_answer(conversation,fill_out_results, answer_database, candidate_templates, private_knowledge_df, synonyms_dictionary):
    start = time.time()
    need_tracking, list_template, confidence_slot_filling_list  = get_template_response(conversation, fill_out_results, candidate_templates, answer_database)
    end = time.time()
    #print("Time of get_template_response fuction: ", end - start)
    if need_tracking == False:
        if len(list_template) == 1:
            #print("template: ", list_template[0])
            return get_answer_based_template(answer_database, list_template[0]), answer_database[list_template[0]]["Product"], confidence_slot_filling_list,list_template[0]
        elif len(list_template) == 2:
            list_answer = [get_answer_based_template(answer_database, template) for template in list_template]
            if len(set(list_answer)) == 1:
                return list_answer[0], answer_database[list_template[0]]["Product"], confidence_slot_filling_list, \
                       list_template[0]
            else:
                response = ""
                list_product = []
                # for index, template in enumerate(list_template):
                #     print("template: ", template)
                #     response += get_answer_based_template(answer_database, template) + "\n\n"
                #     list_product.append(answer_database[template]["Product"].strip())
                # if len(set(list_product)) == 1:
                #     return response, list_product[0], confidence_slot_filling_list, list_template
                # else:
                #     return response, None, confidence_slot_filling_list, list_template
                for index, template in enumerate(list_template):
                    print("template: ", template)
                    response += get_answer_based_template(answer_database, template) + "\n\n"
                    product_value = answer_database[template]["Product"]
                    if isinstance(product_value, str):  # Kiểm tra nếu giá trị là chuỗi
                        list_product.append(product_value.strip())
                    if len(set(list_product)) == 1:
                        return response, list_product[0], confidence_slot_filling_list, list_template
                    else:
                        return response, None, confidence_slot_filling_list, list_template


        elif len(list_template) > 2:
            list_answer = [get_answer_based_template(answer_database, template) for template in list_template]
            if len(set(list_answer)) == 1:
                return list_answer[0], answer_database[list_template[0]]["Product"], confidence_slot_filling_list, list_template[0]
            else:
                response = get_answer_based_template(answer_database, "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                list_product = []
                for index,template in enumerate(list_template):
                    #print("template: ", template)
                    # response += "\t+ " + get_description_based_template(answer_database, template) + "\n\n"
                    response += "\t+ " + str(get_description_based_template(answer_database, template)) + "\n\n"
                    list_product.append(answer_database[template]["Product"])
                if set(list_product) == 1:
                    return response, list_product[0], confidence_slot_filling_list, list_template
                else:
                    return response, None, confidence_slot_filling_list, list_template
        else:
            response = get_answer_based_template(answer_database, "hỏi_lại_khi_không_hiểu|hỏi_lại")
            return response, None, None, ""

    else:
        query_clean = conversation.current_clean_query + " " + processing(conversation.current_product,synonyms_dictionary)
        query = conversation.current_query + " " + processing(conversation.current_product,synonyms_dictionary)
        intent45 = conversation.current_intent
        fill_out_results_tracking = fill_candidate_templates(intent45, candidate_templates, query, query_clean, private_knowledge_df)
        need_tracking2, list_template, confidence_slot_filling_list = get_template_response(conversation, fill_out_results_tracking, candidate_templates,
                                                             answer_database, tracking=True)
        if len(list_template) == 1:
            #print("template: ", list_template[0])
            return get_answer_based_template(answer_database, list_template[0]), answer_database[list_template[0]]["Product"], confidence_slot_filling_list,list_template[0]

        elif len(list_template) == 2:
            list_answer = [get_answer_based_template(answer_database, template) for template in list_template]
            if len(set(list_answer)) == 1:
                return list_answer[0], answer_database[list_template[0]]["Product"], confidence_slot_filling_list, \
                       list_template[0]
            else:
                response = ""
                list_product = []
                for index, template in enumerate(list_template):
                    # print("template: ", template)
                    response += get_answer_based_template(answer_database, template) + "\n\n"
                    list_product.append(answer_database[template]["Product"].strip())
                if len(set(list_product)) == 1:
                    return response, list_product[0], confidence_slot_filling_list, list_template
                else:
                    return response, None, confidence_slot_filling_list, list_template

        else:
            list_answer = [get_answer_based_template(answer_database, template) for template in list_template]
            if len(set(list_answer)) == 1:
                return list_answer[0], answer_database[list_template[0]]["Product"], confidence_slot_filling_list, list_template[0]
            else:
                response = get_answer_based_template(answer_database, "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                list_product = []
                for index,template in enumerate(list_template):
                    #print("template: ", template)
                    des = get_description_based_template(answer_database, template)
                    if str(des) + "\n" not in response:
                    # if des + "\n" not in response:
                        # response +=  "\t+ " + des + "\n"
                        response +=  "\t+ " + str(des) + "\n"
                        list_product.append(answer_database[template]["Product"])
                if len(set(list_product)) == 1:
                    return response, list_product[0], confidence_slot_filling_list, ""
                else:
                    try:
                        return response, None, confidence_slot_filling_list, ""
                    except:
                        response = get_answer_based_template(answer_database, 'hỏi_lại_khi_không_hiểu|hỏi_lại')
                        return response, None, None, ""

def get_product_name_in_query(user_query_clean,list_product,list_product_clean):
    start32344 = time.time()
    outputs = []
    try:
        list_product = list(list_product)
        for index,product in enumerate(list_product_clean):
            if product in user_query_clean:
                outputs.append(list_product[index])
        len_o = [len(item.split(" ")) for item in outputs]
        max_value = max(len_o)
        max_index = len_o.index(max_value)
        end323222 = time.time()
        #print("DEBUG : Time of get_product_name_in_query function: ", end323222 - start32344)
        return outputs[max_index]
    except:
        return ""

def check_product_equal_query(user_query_clean,list_product_clean):
    for index,product in enumerate(list_product_clean):
        if product == user_query_clean:
            return True
    return False

## check ans out domain:
def check_ans_out_domain(question_out_domain_matrix, user_query_clean, laser_model):
    threshold_similarity_product = 0.85
    vector_input1 = laser_model.embed_sentences(user_query_clean, lang='vi')

    cosin_similarities = cos_sim_2d(vector_input1, question_out_domain_matrix)
    max_value = max(cosin_similarities[0])
    max_index = list(cosin_similarities[0]).index(max_value)
    if max_value > threshold_similarity_product:
        return max_index
    else:
        return -1
# Get product name in Description # Lấy tên sản phẩm trong Description
def get_product_based_on_Description(user_query,answer_database):
    for data in answer_database:
        if str(answer_database[data]["Description"]).strip() == str(user_query).strip():
            return answer_database[data]["Product"]
    return ""


class state_conversation:
    def __init__(self,session_id):
        self.session_id = str(session_id)
        my_database = Database(PATH_DATABASE, "database_conservation")
        try:
            last_conservation = my_database.find_last_conservation_user_by_token(session_id)
            self.past_query = last_conservation["query"]
            self.past_clean_query = last_conservation["clean_query"]
            self.past_product = last_conservation["product"]
            self.past_context_error = last_conservation["context_error"]
            self.past_step = last_conservation["step_conversation"]
            self.past_intent = last_conservation["intent"]
            self.current_query = ""
            self.current_clean_query = ""
            self.current_product = ""
            self.current_intent = ""
            self.context_error = self.past_context_error
            self.step_conversation = self.past_step
        except:
            print("TRY FAIL")
            self.past_query = ""
            self.past_clean_query = ""
            self.past_product = ""
            self.past_context_error = ""
            self.past_step = ""
            self.past_intent = ""
            self.current_query = ""
            self.current_clean_query = ""
            self.current_product = ""
            self.current_intent = ""
            self.context_error = 0
            self.step_conversation = 0

    def update_user_query(self, query, processing_query):
        self.current_query = query
        self.current_clean_query = processing_query

    def update_product(self,product):
        self.current_product = product

    def update_context_error(self):
        self.context_error += 1

    #THEM CODE DOT 6:
    def release_context_error(self):
        self.context_error = 0

    def update_step_conversation(self):
        self.step_conversation += 1

    def update_current_intent(self, intent):
        self.current_intent = intent

    def save_database(self):          
        my_database = Database(PATH_DATABASE, "database_conservation")
        my_database.insert_message_user(self.session_id,self.current_query,self.current_clean_query,self.current_product,self.context_error,self.step_conversation,self.current_intent)
        
    def reset(self):
        self.past_query = ""
        self.past_clean_query = ""
        self.past_product = ""
        self.past_context_error = ""
        self.past_step = ""
        self.past_intent = ""
        self.current_query = ""
        self.current_clean_query = ""
        self.current_product = ""
        self.current_intent = ""
        self.context_error = 0
        self.step_conversation = 0


def get_response_api(user_query, session_id,
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
                     THRESHOLD_INTENT_FOR_NOT_TRACKING):

    confidence_answer = 0.0
    confidence_recommendation = 0.0

    response_bot_final = ""
    intent_final = ""
    pattern_template = ""
    question_quality = "good"
    sentiment = "0"
    confidence_score1 = 0
    status = ""
    template =""
    recommendation_response = []
    slot_filling_candidate = []
    slot_filling_result = []

    conversation = state_conversation(session_id)

    # Kiểm tra câu truy vấn có phải câu thiếu dấu không?
    user_query_clean = processing(user_query, synonyms_dictionary)
    if accent_restore.check_have_accent(user_query_clean):
        user_query_clean = user_query_clean.replace("_", " ")
        user_query_clean = accent_restore.remove_accent(user_query_clean)
        user_query_clean = accent_restore.add_accent(user_query_clean)
        user_query = processing(user_query_clean, synonyms_dictionary)
        user_query_clean = user_query
        print("cần thêm dấu")
        print("Accent restore: ", user_query_clean)
    else:
        print(user_query_clean)
    conversation.update_user_query(user_query, user_query_clean)

    index_chatxam = check_available_in_list_chatxam(question_chat_xam_matrix, user_query_clean, laser_model)

    
    print("index_chatxam: ", index_chatxam)
    if user_query_clean.strip() == "":
        print("if 1")
        try:
            language_input = str(detect(user_query))
        except:
            language_input = "vi"

        if language_input != "vi":
            response_bot_final = get_answer_based_template(answer_database_chatxam, 'ngôn_ngữ_truy_vấn')
            confidence_answer = THRESHOLD_RETURN_CONFIDENCE
            confidence_score1 = confidence_answer
            intent_final = "ngôn_ngữ_truy_vấn"
        else:
            response_bot_final = get_answer_based_template(answer_database_chatxam, 'hỏi_đáp_không_ý_nghĩa')
            confidence_answer = THRESHOLD_RETURN_CONFIDENCE
            confidence_score1 = confidence_answer
            intent_final = "hỏi_đáp_không_ý_nghĩa"

    elif str(user_query).replace(" ", "").strip().isnumeric() == False and str(detect(user_query)) != "vi" and len(user_query_clean.split(" ")) > 2 and index_chatxam < 0:
        print("---------------")
        print("if 2")
        print(f"index chatxam: {index_chatxam}")
        print(str(detect(user_query)))
        print("---------------")
        response_bot_final = get_answer_based_template(answer_database_chatxam, 'ngôn_ngữ_truy_vấn')
        confidence_answer = THRESHOLD_RETURN_CONFIDENCE
        confidence_score1 = confidence_answer
        intent_final = "ngôn_ngữ_truy_vấn"

    elif len(user_query_clean.split(" ")) == 1 or str(user_query).replace(" ", "").strip().isnumeric() == True:
        print("if 3")
        if str(user_query).replace(" ", "").strip().isnumeric() == True:
            if conversation.context_error < THRESHOLD_ERROR_FOR_AGENT:
                response_bot_final = get_answer_based_template(answer_database, 'hỏi_lại_khi_không_hiểu|hỏi_lại')
                confidence_answer = THRESHOLD_RETURN_CONFIDENCE
                confidence_score1 = confidence_answer
                status = "askagain"
                intent_final = "hỏi_lại_khi_không_hiểu"
            else:
                response_bot_final = get_answer_based_template(answer_database,"chuyển_hội_thoại_cho_agent|agent" )
                confidence_answer = THRESHOLD_RETURN_CONFIDENCE
                confidence_score1 = confidence_answer
                status = "unknown"
                intent_final = "chuyển_hội_thoại_cho_agent"
                conversation.release_context_error()
        elif index_chatxam < 0:
            response_bot_final = get_answer_based_template(answer_database_chatxam, 'hỏi_đáp_không_ý_nghĩa')
            confidence_answer = THRESHOLD_RETURN_CONFIDENCE
            confidence_score1 = confidence_answer
            intent_final = "hỏi_đáp_không_ý_nghĩa"
        else:
            template = template_chatxam[index_chatxam]
            response_bot_final = get_answer_based_template(answer_database_chatxam, template)
            confidence_answer = THRESHOLD_RETURN_CONFIDENCE
            confidence_score1 = confidence_answer
            intent_final = template

    elif user_query_clean in description_clean:
        print("if 4")
        index = description_clean.index(user_query_clean)
        response_bot_final = get_answer_based_template(answer_database, answer_template[index])
        confidence_answer = 1.0
        confidence_score1 = confidence_answer
        intent_final = "Description"
        conversation.update_step_conversation()
        product = get_product_based_on_Description(description[index], answer_database)
        conversation.update_product(product)
        

    elif check_product_equal_query(user_query_clean, list_product_clean) == True:
        print("if 5")
        if conversation.past_query.strip() != "" and user_query_clean != conversation.past_clean_query and user_query_clean != conversation.current_product:
            query_clean = conversation.past_clean_query.strip() + " " + user_query_clean.strip()
            query_clean = query_clean.replace("_", " ")
            if len(clean_text(conversation.past_product)) != 0:
                query_clean = query_clean.replace(clean_text(conversation.past_product), " ")
            else: 
                pass
            intent1, confidence_score1, confidence_index1, intent2, confidence_score2, confidence_index2 = intent_classification(
                query_clean, intent_model, vectorizer, list_label, max_length)
            print(confidence_score1)
            if confidence_score1 >= THRESHOLD_INTENT_CONFIDENCE_DEFAULT:
                conversation.update_current_intent(intent1)
                candidate_templates = extract_template_from_database(intent1, answer_template)
                fill_out_results = fill_candidate_templates(intent1, candidate_templates, query_clean,
                                                            query_clean, private_knowledge_df)

                response, product, confidence_slot_filling_list, template_response = bot_answer(
                    conversation, fill_out_results, answer_database, candidate_templates,
                    private_knowledge_df, synonyms_dictionary)

                slot_filling_candidate = candidate_templates
                slot_filling_result = fill_out_results
                pattern_template = template_response
                if product != None:
                    conversation.update_product(product)
                    # print("Vi tri 4")
                    response_bot_final = response
                    intent_final = intent1
                    # Calculuate confidence answer score.
                    if len(confidence_slot_filling_list) == 1:
                        confidence_answer = (confidence_score1 + confidence_slot_filling_list[0]) / 2

                    elif len(confidence_slot_filling_list) == 2:
                        confidence_answer = (confidence_score1 + confidence_slot_filling_list[0] +
                                             confidence_slot_filling_list[1]) / 3
                    else:
                        sum_slot = sum(confidence_slot_filling_list) / len(confidence_slot_filling_list)
                        confidence_recommendation = (confidence_score1 + sum_slot) / 2
                else:
                    suggestions, recommendation_score = get_recommendation(user_query,intent1, dict_description,
                                                                           dict_description_preprocessing,
                                                                           dict_description_matrix, laser_model,
                                                                           answer_template,
                                                                           synonyms_dictionary,
                                                                           topk=NUMBER_OF_RECOMMENDATION_RESULT)
                    if len(suggestions) > 1:
                        response_bot = get_answer_based_template(answer_database,"đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                        for i, item in enumerate(suggestions):
                            response_bot += "\t+ " + item + "\n"
                            recommendation_response.append(item)
                        response_bot = response_bot[:-1]
                        response_bot_final = response_bot
                        intent_final = intent1
                        confidence_recommendation = recommendation_score
                        status = "recommend"
                    else:
                        response_bot_final = response
                        intent_final = intent1
                        confidence_recommendation = recommendation_score
                        status = "recommend"
                    try:
                        product = get_product_name_in_query(user_query_clean, list_product,
                                                            list_product_clean)
                        conversation.update_product(product)
                    except:
                        conversation.update_product(" ")
            else:
                print("vị trí là gì")
                query_clean =  user_query_clean + " là gì"
                intent1, confidence_score1, confidence_index1, intent2, confidence_score2, confidence_index2 = intent_classification(
                    query_clean, intent_model, vectorizer, list_label, max_length)

                if confidence_score1 >= THRESHOLD_INTENT_CONFIDENCE_DEFAULT:
                    conversation.update_current_intent(intent1)
                    candidate_templates = extract_template_from_database(intent1, answer_template)
                    fill_out_results = fill_candidate_templates(intent1, candidate_templates, query_clean,
                                                                query_clean, private_knowledge_df)
                    response, product, confidence_slot_filling_list, template_response = bot_answer(
                        conversation, fill_out_results, answer_database, candidate_templates,
                        private_knowledge_df, synonyms_dictionary)
                    
                    slot_filling_candidate = candidate_templates
                    slot_filling_result = fill_out_results
                    pattern_template = template_response
                    if product != None:
                        conversation.update_product(product)
                        # print("Vi tri 4")
                        response_bot_final = response
                        intent_final = intent1
                        # Calculuate confidence answer score.
                        if len(confidence_slot_filling_list) == 1:
                            confidence_answer = (confidence_score1 + confidence_slot_filling_list[0]) / 2

                        elif len(confidence_slot_filling_list) == 2:
                            confidence_answer = (confidence_score1 + confidence_slot_filling_list[0] +
                                                 confidence_slot_filling_list[1]) / 3
                        else:
                            sum_slot = sum(confidence_slot_filling_list) / len(confidence_slot_filling_list)
                            confidence_recommendation = (confidence_score1 + sum_slot) / 2
                    else:
                        suggestions, recommendation_score = get_recommendation(user_query,intent1, dict_description,
                                                                           dict_description_preprocessing,
                                                                           dict_description_matrix, laser_model,
                                                                           answer_template,
                                                                           synonyms_dictionary,
                                                                           topk=NUMBER_OF_RECOMMENDATION_RESULT)
                        if len(suggestions) > 1:
                            response_bot = get_answer_based_template(answer_database,
                                                                     "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                            for i, item in enumerate(suggestions):
                                response_bot += "\t+ " + item + "\n"
                                recommendation_response.append(item)
                            response_bot = response_bot[:-1]
                            response_bot_final = response_bot
                            intent_final = intent1
                            confidence_recommendation = recommendation_score
                            status = "recommend"
                        else:
                            response_bot_final = response
                            intent_final = intent1
                            confidence_recommendation = recommendation_score
                            status = "recommend"
                        try:
                            product = get_product_name_in_query(user_query_clean, list_product,
                                                                list_product_clean)
                            conversation.update_product(product)
                        except:
                            conversation.update_product(" ")
        else:
            query_clean = user_query_clean + " là gì"
            intent1, confidence_score1, confidence_index1, intent2, confidence_score2, confidence_index2 = intent_classification(
                query_clean, intent_model, vectorizer, list_label, max_length)

            if confidence_score1 >= THRESHOLD_INTENT_CONFIDENCE_DEFAULT:
                conversation.update_current_intent(intent1)
                candidate_templates = extract_template_from_database(intent1, answer_template)
                # print("candidate_templates")
                # print(candidate_templates)
                fill_out_results = fill_candidate_templates(intent1, candidate_templates, query_clean,
                                                            query_clean, private_knowledge_df)
                # print("fill out result")
                # print(fill_out_results)
                response, product, confidence_slot_filling_list, template_response = bot_answer(
                    conversation, fill_out_results, answer_database, candidate_templates,
                    private_knowledge_df, synonyms_dictionary)

                slot_filling_candidate = candidate_templates
                slot_filling_result = fill_out_results
                pattern_template = template_response
                if product != None:
                    conversation.update_product(product)
                    # print("Vi tri 4")
                    response_bot_final = response
                    intent_final = intent1
                    # Calculuate confidence answer score.
                    if len(confidence_slot_filling_list) == 1:
                        confidence_answer = (confidence_score1 + confidence_slot_filling_list[0]) / 2

                    elif len(confidence_slot_filling_list) == 2:
                        confidence_answer = (confidence_score1 + confidence_slot_filling_list[0] +
                                             confidence_slot_filling_list[1]) / 3
                    else:
                        sum_slot = sum(confidence_slot_filling_list) / len(confidence_slot_filling_list)
                        confidence_recommendation = (confidence_score1 + sum_slot) / 2
                else:
                    suggestions, recommendation_score = get_recommendation(user_query,intent1, dict_description,
                                                                           dict_description_preprocessing,
                                                                           dict_description_matrix, laser_model,
                                                                           answer_template,
                                                                           synonyms_dictionary,
                                                                           topk=NUMBER_OF_RECOMMENDATION_RESULT)
                    if len(suggestions) > 1:
                        response_bot = get_answer_based_template(answer_database,
                                                                 "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                        for i, item in enumerate(suggestions):
                            response_bot += "\t+ " + item + "\n"
                            recommendation_response.append(item)
                        response_bot = response_bot[:-1]
                        response_bot_final = response_bot
                        intent_final = intent1
                        confidence_recommendation = recommendation_score
                        status = "recommend"
                    else:
                        response_bot_final = response
                        intent_final = intent1
                        confidence_recommendation = recommendation_score
                        status = "recommend"
                    try:
                        product = get_product_name_in_query(user_query_clean, list_product,
                                                            list_product_clean)
                        conversation.update_product(product)
                    except:
                        conversation.update_product(" ")

    # Check câu trong chat xam
    elif index_chatxam > 1:
        print("if 6")
        template = template_chatxam[index_chatxam]
        print("template: ", template)
        response_bot_final = get_answer_based_template(answer_database_chatxam, template)
        confidence_answer = THRESHOLD_RETURN_CONFIDENCE
        confidence_score1 = confidence_answer
        intent_final = template

    elif predict_sent(user_query_clean, Out_intent_model, Out_vectorizer,Out_max_length) == 0:
        print("if 6")
        index_chat_outdomain = check_ans_out_domain(question_out_domain_matrix, user_query_clean, laser_model)
        if index_chat_outdomain>=0:
            response_bot_final = list_ans_out_domain[index_chat_outdomain]
            confidence_answer = THRESHOLD_RETURN_CONFIDENCE
            confidence_score1 = confidence_answer
            intent_final = "truy_vấn_ngoài_phạm_vi"
        else:     
            response_bot_final = get_answer_based_template(answer_database_chatxam, 'truy_vấn_ngoài_phạm_vi')
            confidence_answer = THRESHOLD_RETURN_CONFIDENCE
            confidence_score1 = confidence_answer
            intent_final = "truy_vấn_ngoài_phạm_vi"
    else:
        print("if 7")
        print(f"user_query_clean: {user_query_clean}")
        intent1, confidence_score1, confidence_index1, intent2, confidence_score2, confidence_index2 = intent_classification(
        user_query_clean, intent_model, vectorizer, list_label, max_length)
        conversation.update_step_conversation()
        if confidence_score1 < THRESHOLD_INTENT_FOR_NOT_TRACKING and check_similarity_product_query(
                user_query_clean, list_product_clean, product_name_matrix, laser_model) == False:
            conversation.update_context_error()
            response_bot_final = get_answer_based_template(answer_database, 'hỏi_lại_khi_không_hiểu|hỏi_lại')
            confidence_answer = THRESHOLD_RETURN_CONFIDENCE
            status = "askagain"
            intent_final = "hỏi_lại_khi_không_hiểu"
        else:
               # conversation.update_user_query(user_query, user_query_clean)
            # Trường hợp này nếu bộ classifier xác định được ý giá trị confidence trên query người dùng. Sẽ tiến hành xác định câu trả lời
            print("confident of if 8:", confidence_score1)
            if confidence_score1 >= THRESHOLD_INTENT_CONFIDENCE_DEFAULT:
                # Sau khi xác định confidence nhiều thì phải kiểm tra xem là có product trong đó không?
                # Nếu có product trong đó thì xử lý tìm câu trả lời bình thường
                if check_similarity_product_query(user_query_clean, list_product_clean, product_name_matrix,
                                                  laser_model) == True and confidence_score1 > THRESHOLD_INTENT_CONFIDENCE_DEFAULT + 0.1:
                    print("có product trong query và xử lý bình thường")
                    # LOI SAIIII
                    conversation.update_current_intent(intent1)
                    candidate_templates = extract_template_from_database(intent1, answer_template)
                    fill_out_results = fill_candidate_templates(intent1, candidate_templates, user_query,
                                                                user_query_clean, private_knowledge_df)

                    slot_filling_candidate = candidate_templates
                    slot_filling_result = fill_out_results
                    # print("conversation product: ", conversation.current_product )
                    start323 = time.time()
                    response, product, confidence_slot_filling_list, template_response = bot_answer(conversation,
                                                                                                    fill_out_results,
                                                                                                    answer_database,
                                                                                                    candidate_templates,
                                                                                                    private_knowledge_df,
                                                                                                    synonyms_dictionary)
                    end3232 = time.time()
                    # print("Time of bot_answer: ", end3232 - start323)
                    pattern_template = template_response

                    if product != None:
                        conversation.update_product(product)
                        # print("Vi tri 1")
                        response_bot_final = response
                        intent_final = intent1
                        # Calculuate confidence answer score.
                        if len(confidence_slot_filling_list) == 1:
                            confidence_answer = (confidence_score1 + confidence_slot_filling_list[0]) / 2
                        elif len(confidence_slot_filling_list) == 2:
                            confidence_answer = (confidence_score1 + confidence_slot_filling_list[0] +
                                                 confidence_slot_filling_list[1]) / 3
                        else:
                            sum_slot = sum(confidence_slot_filling_list) / len(confidence_slot_filling_list)
                            confidence_recommendation = (confidence_score1 + sum_slot) / 2
                    else:
                        suggestions, recommendation_score = get_recommendation(user_query,intent1, dict_description,
                                                                           dict_description_preprocessing,
                                                                           dict_description_matrix, laser_model,
                                                                           answer_template,
                                                                           synonyms_dictionary,
                                                                           topk=NUMBER_OF_RECOMMENDATION_RESULT)
                        if len(suggestions) > 1:
                            response_bot = get_answer_based_template(answer_database,
                                                                     "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                            # conversation.update_context_error()
                            for i, item in enumerate(suggestions):
                                response_bot += "\t+ " + item + "\n"
                                recommendation_response.append(item)
                            response_bot = response_bot[:-1]
                            # print("Vi tri 2")
                            response_bot_final = response_bot
                            intent_final = intent1
                            confidence_recommendation = recommendation_score
                            status = "recommend"
                        else:
                            # print("Vi tri 3")
                            response_bot_final = response
                            intent_final = intent1
                            confidence_recommendation = recommendation_score
                            status = "recommend"
                        try:
                            product = get_product_name_in_query(user_query_clean, list_product, list_product_clean)
                            # print("Product in query: ", product)
                            conversation.update_product(product)
                        except:
                            conversation.update_product(" ")

                # Trường hợp trong câu không có product thì phải thêm product vào để fill cho hợp lý.
                else:
                    print("KHông có product trong câu query")
                    # print("MO DAU HET 4: conversation.current_product", conversation.current_product)
                    # print("MO DAU HET 4: conversation.past_product", conversation.past_product)
                    if conversation.past_product != None:
                        query = user_query + " " + str(conversation.past_product)
                    else:
                        conversation.past_product = ""
                    query_clean = user_query_clean + " " + processing(conversation.past_product,
                                                                      synonyms_dictionary)
                    intent1, confidence_score1, confidence_index1, intent2, confidence_score2, confidence_index2 = intent_classification(
                        query_clean, intent_model, vectorizer, list_label, max_length)
                    #print("RAW INTENT CONFIDENCE SCORE 5", intent1, confidence_score1)
                    # Sau khi tracking mà giá trị confidence score lớn hơn ngưỡng thì tiến hành tìm câu trả lời
                    if confidence_score1 >= THRESHOLD_INTENT_CONFIDENCE_TRACKING:
                        conversation.update_current_intent(intent1)
                        candidate_templates = extract_template_from_database(intent1, answer_template)
                        fill_out_results = fill_candidate_templates(intent1, candidate_templates, query,
                                                                    query_clean, private_knowledge_df)

                        # start323 = time.time()
                        response, product, confidence_slot_filling_list, template_response = bot_answer(
                            conversation, fill_out_results, answer_database, candidate_templates,
                            private_knowledge_df, synonyms_dictionary)

                        # end3232 = time.time()
                        # print("Time of bot_answer: ", end3232 - start323)

                        slot_filling_candidate = candidate_templates
                        slot_filling_result = fill_out_results

                        pattern_template = template_response
                        if product != None:
                            conversation.update_product(product)
                            # print("Vi tri 4")
                            response_bot_final = response
                            intent_final = intent1
                            # Calculuate confidence answer score.
                            if len(confidence_slot_filling_list) == 1:
                                confidence_answer = (confidence_score1 + confidence_slot_filling_list[0]) / 2

                            elif len(confidence_slot_filling_list) == 2:
                                confidence_answer = (confidence_score1 + confidence_slot_filling_list[0] +
                                                     confidence_slot_filling_list[1]) / 3
                            else:
                                sum_slot = sum(confidence_slot_filling_list) / len(confidence_slot_filling_list)
                                confidence_recommendation = (confidence_score1 + sum_slot) / 2
                        else:
                            suggestions, recommendation_score = get_recommendation(user_query,intent1, dict_description,
                                                                           dict_description_preprocessing,
                                                                           dict_description_matrix, laser_model,
                                                                           answer_template,
                                                                           synonyms_dictionary,
                                                                           topk=NUMBER_OF_RECOMMENDATION_RESULT)
                            if len(suggestions) > 1:
                                # conversation.update_context_error()
                                response_bot = get_answer_based_template(answer_database,
                                                                         "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                                for i, item in enumerate(suggestions):
                                    response_bot += "\t+ " + item + "\n"
                                    recommendation_response.append(item)
                                response_bot = response_bot[:-1]
                                # print("Vi tri 5")
                                response_bot_final = response_bot
                                intent_final = intent1
                                confidence_recommendation = recommendation_score
                                status = "recommend"
                            else:
                                # print("Vi tri 6")
                                response_bot_final = response
                                intent_final = intent1
                                confidence_recommendation = recommendation_score
                                status = "recommend"
                            try:
                                product = get_product_name_in_query(user_query_clean, list_product,
                                                                    list_product_clean)
                                # print("Product in query: ", product)
                                conversation.update_product(product)
                            except:
                                conversation.update_product(" ")

                    else:
                        suggestions, recommendation_score = get_recommendation(user_query,intent1, dict_description,
                                                                           dict_description_preprocessing,
                                                                           dict_description_matrix, laser_model,
                                                                           answer_template,
                                                                           synonyms_dictionary,
                                                                           topk=NUMBER_OF_RECOMMENDATION_RESULT)
                        if len(suggestions) > 1:
                            # conversation.update_context_error()
                            response_bot = get_answer_based_template(answer_database,
                                                                     "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                            for i, item in enumerate(suggestions):
                                response_bot += "\t+ " + item + "\n"
                                recommendation_response.append(item)
                            response_bot = response_bot[:-1]
                            # print("Vi tri 7")
                            intent_final = intent1
                            response_bot_final = response_bot
                            confidence_recommendation = recommendation_score
                            status = "recommend"
                        elif len(suggestions) == 1:
                            response_bot = get_answer_based_template(answer_database,
                                                                     "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                            response_bot += "\t+ " + suggestions[0] + "\n"
                            recommendation_response.append(suggestions[0])
                            response_bot = response_bot[:-1]
                            # print("Vi tri 8")
                            intent_final = intent1
                            response_bot_final = response_bot
                            confidence_recommendation = recommendation_score
                            status = "recommend"
            # Ngược lại giá trị confidence score thấp thì sẽ kiểm tra các điều kiện như sau:
            else:
                # Đầu tiên là kiểm tra xem trong câu truy vấn có chứa sản phẩm hay không. Nếu có thì sẽ tracking lại câu trước để dự đoán.
                print("Đầu tiên là kiểm tra xem trong câu truy vấn có chứa sản phẩm hay không. Nếu có thì sẽ tracking lại câu trước để dự đoán.")
                if check_similarity_product_query(user_query, list_product_clean, product_name_matrix, laser_model) == True:
                    print("Có product trong câu query")
                    product_in_query = get_product_name_in_query(user_query_clean, list_product, list_product_clean)
                    #print("product_in_query: ", product_in_query)
                    # Khi xác định trong truy vấn đang nhắc đến một sản phẩm. Thì ta cần xác định xem product này có giống như product nhắc trước đó không
                    if product_in_query == conversation.current_product:  
                        ### TRACKING
                        print("có product nhưng confident thấp")
                        # Nếu product trong bình luận giống product hiện tại. mà giá trị confidence vẫn thấp ==> suggestion hoặc thông báo không hiểu
                        suggestions, recommendation_score = get_recommendation(user_query,intent1, dict_description,
                                                                           dict_description_preprocessing,
                                                                           dict_description_matrix, laser_model,
                                                                           answer_template,
                                                                           synonyms_dictionary,
                                                                           topk=NUMBER_OF_RECOMMENDATION_RESULT)
                        if len(suggestions) > 1:
                            # conversation.update_context_error()
                            response_bot = get_answer_based_template(answer_database,
                                                                     "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                            for i, item in enumerate(suggestions):
                                response_bot += "\t+ " + item + "\n"
                                recommendation_response.append(item)
                            response_bot = response_bot[:-1]
                            # print("Vi tri 7")
                            response_bot_final = response_bot
                            intent_final = intent1
                            confidence_recommendation = recommendation_score
                            status = "recommend"
                        elif len(suggestions) == 1:
                            response_bot = get_answer_based_template(answer_database,
                                                                     "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                            response_bot += "\t+ " + suggestions[0] + "\n"
                            recommendation_response.append(suggestions[0])
                            response_bot = response_bot[:-1]
                            # print("Vi tri 8")
                            response_bot_final = response_bot
                            intent_final = intent1
                            confidence_recommendation = recommendation_score
                            status = "recommend"
                    # Trường hợp: Nếu product trong truy vấn khác với product hiện tại thì thử tracking về câu query phía trên + câu này để xác định thử confidence score
                    else:
                        if confidence_score1 < THRESHOLD_INTENT_CONFIDENCE_TRACKING:
                            if conversation.current_product != conversation.past_product:
                                query = conversation.past_query.replace(conversation.past_product, " ") + " " + conversation.current_query
                                print("query tracking:",query)
                                if "_" not in query:
                                    query_clean = processing(conversation.past_query.replace(conversation.past_product, " "),
                                                             synonyms_dictionary) + " " + processing(
                                        conversation.current_query, synonyms_dictionary)
                                else:
                                    query_clean = query
                            intent1, confidence_score1, confidence_index1, intent2, confidence_score2, confidence_index2 = intent_classification(
                                query_clean, intent_model, vectorizer, list_label, max_length)
                            #print("RAW INTENT CONFIDENCE SCORE 6", intent1, confidence_score1)
                            # Sau khi tracking mà giá trị confidence score lớn hơn ngưỡng thì tiến hành tìm câu trả lời
                            if confidence_score1 >= THRESHOLD_INTENT_CONFIDENCE_TRACKING:
                                conversation.update_current_intent(intent1)
                                candidate_templates = extract_template_from_database(intent1, answer_template)
                                fill_out_results = fill_candidate_templates(intent1, candidate_templates, query,
                                                                            query_clean, private_knowledge_df)

                                slot_filling_candidate = candidate_templates
                                slot_filling_result = fill_out_results

                                response, product, confidence_slot_filling_list, template_response = bot_answer(
                                    conversation, fill_out_results, answer_database,
                                    candidate_templates, private_knowledge_df, synonyms_dictionary)

                                pattern_template = template_response
                                # print("Vi tri 9")

                                if product != None:
                                    conversation.update_product(product)
                                    product_final = product
                                    # print("Vi tri 9.1")
                                    response_bot_final = response
                                    intent_final = intent1
                                    # Calculuate confidence answer score.
                                    if len(confidence_slot_filling_list) == 1:
                                        confidence_answer = (confidence_score1 + confidence_slot_filling_list[
                                            0]) / 2

                                    elif len(confidence_slot_filling_list) == 2:
                                        confidence_answer = (confidence_score1 + confidence_slot_filling_list[0] +
                                                             confidence_slot_filling_list[1]) / 3

                                    else:
                                        sum_slot = sum(confidence_slot_filling_list) / len(
                                            confidence_slot_filling_list)
                                        confidence_recommendation = (confidence_score1 + sum_slot) / 2
                                else:
                                    # conversation.update_context_error()
                                    suggestions, recommendation_score = get_recommendation(user_query,intent1, dict_description,
                                                                           dict_description_preprocessing,
                                                                           dict_description_matrix, laser_model,
                                                                           answer_template,
                                                                           synonyms_dictionary,
                                                                           topk=NUMBER_OF_RECOMMENDATION_RESULT)
                                    if len(suggestions) > 1:
                                        response_bot = get_answer_based_template(answer_database,
                                                                                 "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                                        for i, item in enumerate(suggestions):
                                            response_bot += "\t+ " + item + "\n"
                                            recommendation_response.append(item)
                                        response_bot = response_bot[:-1]
                                        # print("Vi tri 9.2")
                                        response_bot_final = response_bot
                                        intent_final = intent1
                                        confidence_recommendation = recommendation_score
                                        status = "recommend"
                                    else:
                                        # print("Vi tri 9.3")
                                        response_bot_final = response
                                        intent_final = intent1
                                        confidence_recommendation = recommendation_score
                                        status = "recommend"
                                    try:
                                        product = get_product_name_in_query(user_query_clean, list_product,
                                                                            list_product_clean)
                                        # print("Product in query: ", product)
                                        conversation.update_product(product)
                                    except:
                                        conversation.update_product(" ")

                            # Nếu tracking vẫn không tìm thấy ngưỡng nữa thì ==> suggestion hoặc thông báo không hiểu
                            else:
                                suggestions, recommendation_score = get_recommendation(user_query,intent1, dict_description,
                                                                           dict_description_preprocessing,
                                                                           dict_description_matrix, laser_model,
                                                                           answer_template,
                                                                           synonyms_dictionary,
                                                                           topk=NUMBER_OF_RECOMMENDATION_RESULT)
                                if len(suggestions) != 0:
                                    if len(suggestions) > 1:
                                        # conversation.update_context_error()
                                        response_bot = get_answer_based_template(answer_database,
                                                                                 "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                                        for i, item in enumerate(suggestions):
                                            response_bot += "\t+ " + item + "\n"
                                            recommendation_response.append(item)
                                        response_bot = response_bot[:-1]
                                        # print("Vi tri 10")
                                        response_bot_final = response_bot
                                        intent_final = intent1
                                        confidence_recommendation = recommendation_score
                                        status = "recommend"
                                    else:
                                        response_bot = get_answer_based_template(answer_database,
                                                                                 "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                                        response_bot += "\t+ " + suggestions[0] + "\n"
                                        recommendation_response.append(suggestions[0])
                                        response_bot = response_bot[:-1]
                                        # print("Vi tri 11")
                                        response_bot_final = response_bot
                                        intent_final = intent1
                                        confidence_recommendation = recommendation_score
                                        status = "recommend"
                                else:
                                    conversation.update_context_error()
                                    if conversation.context_error < THRESHOLD_ERROR_FOR_AGENT:
                                        response_bot_final = answer_database["hỏi_lại_khi_không_hiểu|hỏi_lại"][
                                            "Answer"]
                                        confidence_answer = THRESHOLD_RETURN_CONFIDENCE
                                        intent_final = "hỏi_lại_khi_không_hiểu"
                                        status = "askagain"
                                    else:
                                        response_bot_final = answer_database["chuyển_hội_thoại_cho_agent|agent"][
                                            "Answer"]
                                        confidence_answer = 0.0
                                        intent_final = "chuyển_hội_thoại_cho_agent"
                                        status = "unknown"
                                        conversation.release_context_error()
                                # Lấy name product ở trong câu này coi có không
                                try:
                                    product = get_product_name_in_query(user_query_clean, list_product,
                                                                        list_product_clean)
                                    # print("Product in query: ", product)
                                    conversation.update_product(product)
                                except:
                                    pass

                        else:
                            suggestions, recommendation_score = get_recommendation(user_query,intent1, dict_description,
                                                                           dict_description_preprocessing,
                                                                           dict_description_matrix, laser_model,
                                                                           answer_template,
                                                                           synonyms_dictionary,
                                                                           topk=NUMBER_OF_RECOMMENDATION_RESULT)
                            if len(suggestions) > 1:
                                # conversation.update_context_error()
                                response_bot = get_answer_based_template(answer_database,
                                                                         "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                                for i, item in enumerate(suggestions):
                                    response_bot += "\t+ " + item + "\n"
                                    recommendation_response.append(item)
                                response_bot = response_bot[:-1]
                                # print("Vi tri 12")
                                response_bot_final = response_bot
                                intent_final = intent1
                                confidence_recommendation = recommendation_score
                                status = "recommend"
                            elif len(suggestions) == 1:
                                response_bot = get_answer_based_template(answer_database,
                                                                         "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                                response_bot += "\t+ " + suggestions[0] + "\n"
                                recommendation_response.append(suggestions[0])
                                response_bot = response_bot[:-1]
                                # print("Vi tri 13")
                                response_bot_final = response_bot
                                intent_final = intent1
                                confidence_recommendation = recommendation_score
                                status = "recommend"
                            else:
                                if conversation.context_error >= THRESHOLD_ERROR_FOR_AGENT:
                                    conversation.release_context_error()
                                    response_bot_final = get_answer_based_template(answer_database,
                                                                                   'chuyển_hội_thoại_cho_agent|agent')
                                    confidence_answer = 0.0
                                    intent_final = "chuyển_hội_thoại_cho_agent"
                                    status = "unknown"
                                else:
                                    response_bot_final = get_answer_based_template(answer_database,
                                                                                   'hỏi_lại_khi_không_hiểu|hỏi_lại')
                                    confidence_answer = THRESHOLD_RETURN_CONFIDENCE
                                    intent_final = "hỏi_lại_khi_không_hiểu"
                                    status = "askagain"
                                    conversation.update_context_error()
                                    

                        # Cập nhật cái product vào product hiện tại
                        if product_in_query.strip() != "":
                            conversation.update_product(product_in_query)  # Nếu có thì cập nhật vào hiện tại.

                # Trường hợp này là trường hợp hợp trong quy vấn không có đề cập đến product. Thì kiểm tra các điều kiện sau
                else:
                    ### TRACKING 
                    print("KHông có product")
                    # Nếu hiện tại không có product thì sử dụng ==> suggestion hoặc thông báo không hiểu
                    if conversation.current_product.strip() == "":
                        suggestions, recommendation_score = get_recommendation(user_query,intent1, dict_description,
                                                                           dict_description_preprocessing,
                                                                           dict_description_matrix, laser_model,
                                                                           answer_template,
                                                                           synonyms_dictionary,
                                                                           topk=NUMBER_OF_RECOMMENDATION_RESULT)
                        if len(suggestions) > 1:
                            # conversation.update_context_error()
                            response_bot = get_answer_based_template(answer_database,
                                                                     "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                            for i, item in enumerate(suggestions):
                                response_bot += "\t+ " + item + "\n"
                                recommendation_response.append(item)
                            response_bot = response_bot[:-1]
                            # print("Vi tri 12")
                            response_bot_final = response_bot
                            intent_final = intent1
                            confidence_recommendation = recommendation_score
                            status = "recommend"
                        elif len(suggestions) == 1:
                            response_bot = "Có phải bạn muốn hỏi ý định này không: \n"
                            response_bot += "\t+ " + suggestions[0] + "\n"
                            recommendation_response.append(suggestions[0])
                            response_bot = response_bot[:-1]
                            # print("Vi tri 13")
                            response_bot_final = response_bot
                            intent_final = intent1
                            confidence_recommendation = recommendation_score
                            status = "recommend"
                        else:
                            
                            if conversation.context_error < THRESHOLD_ERROR_FOR_AGENT:
                                conversation.update_context_error()
                                response_bot_final = get_answer_based_template(answer_database,
                                                                               'hỏi_lại_khi_không_hiểu|hỏi_lại')
                                confidence_answer = THRESHOLD_RETURN_CONFIDENCE
                                status = "askagain"
                                intent_final = "hỏi_lại_khi_không_hiểu"
                            else:
                                conversation.release_context_error()
                                response_bot_final = get_answer_based_template(answer_database,
                                                                               'chuyển_hội_thoại_cho_agent|agent')
                                confidence_answer = 0.0
                                status = "unknown"
                                intent_final = "chuyển_hội_thoại_cho_agent"
                    # Trường hợp trường hợp tồn tại 1 product đang trong nội dung cuộc nói chuyện ==> tracking kiểm tra thử
                    else:
                        ### TRACKING
                        query = conversation.current_query + " " + conversation.current_product
                        print("Query có product:", query)
                        query_clean = conversation.current_clean_query + " " + processing(
                            conversation.current_product, synonyms_dictionary)
                        intent1, confidence_score1, confidence_index1, intent2, confidence_score2, confidence_index2 = intent_classification(
                            query_clean, intent_model, vectorizer, list_label, max_length)
                        #print("RAW INTENT CONFIDENCE SCORE 7", intent1, confidence_score1)
                        if confidence_score1 >= THRESHOLD_INTENT_CONFIDENCE_TRACKING:
                            conversation.update_current_intent(intent1)
                            candidate_templates = extract_template_from_database(intent1, answer_template)
                            fill_out_results = fill_candidate_templates(intent1, candidate_templates, query,
                                                                        query_clean, private_knowledge_df)
                            slot_filling_candidate = candidate_templates
                            slot_filling_result = fill_out_results

                            response, product, confidence_slot_filling_list, template_response = bot_answer(
                                conversation, fill_out_results, answer_database, candidate_templates,
                                private_knowledge_df, synonyms_dictionary)

                            pattern_template = template_response
                            # print("Vi tri 16")

                            if product != None:
                                conversation.update_product(product)
                                # print("Vi tri 16.1")
                                response_bot_final = response
                                intent_final = intent1
                                # Calculuate confidence answer score.
                                if len(confidence_slot_filling_list) == 1:
                                    confidence_answer = (confidence_score1 + confidence_slot_filling_list[
                                        0]) / 2

                                elif len(confidence_slot_filling_list) == 2:
                                    confidence_answer = (confidence_score1 + confidence_slot_filling_list[0] +
                                                         confidence_slot_filling_list[1]) / 3

                                else:
                                    sum_slot = sum(confidence_slot_filling_list) / len(confidence_slot_filling_list)
                                    confidence_recommendation = (confidence_score1 + sum_slot) / 2

                            else:
                                suggestions, recommendation_score = get_recommendation(user_query,intent1, dict_description,
                                                                           dict_description_preprocessing,
                                                                           dict_description_matrix, laser_model,
                                                                           answer_template,
                                                                           synonyms_dictionary,
                                                                           topk=NUMBER_OF_RECOMMENDATION_RESULT)
                                if len(suggestions) > 1:
                                    # conversation.update_context_error()
                                    response_bot = get_answer_based_template(answer_database,
                                                                             "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                                    for i, item in enumerate(suggestions):
                                        response_bot += "\t+ " + item + "\n"
                                        recommendation_response.append(item)
                                    response_bot = response_bot[:-1]
                                    # print("Vi tri 16.2")
                                    response_bot_final = response_bot
                                    intent_final = intent1
                                    confidence_recommendation = recommendation_score
                                    status = "recommend"
                                else:
                                    # print("Vi tri 16.3")
                                    response_bot_final = response
                                    intent_final = intent1
                                    confidence_recommendation = recommendation_score
                                    status = "recommend"
                                try:
                                    product = get_product_name_in_query(user_query_clean, list_product,
                                                                        list_product_clean)
                                    # print("Product in query: ", product)
                                    conversation.update_product(product)
                                except:
                                    conversation.update_product(" ")

                        else:
                            suggestions, recommendation_score = get_recommendation(user_query,intent1, dict_description,
                                                                           dict_description_preprocessing,
                                                                           dict_description_matrix, laser_model,
                                                                           answer_template,
                                                                           synonyms_dictionary,
                                                                           topk=NUMBER_OF_RECOMMENDATION_RESULT)
                            if len(suggestions) != 0:
                                if len(suggestions) > 1:
                                    # conversation.update_context_error()
                                    response_bot = get_answer_based_template(answer_database,
                                                                             "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                                    for i, item in enumerate(suggestions):
                                        response_bot += "\t+ " + item + "\n"
                                        recommendation_response.append(item)
                                    response_bot = response_bot[:-1]
                                    # print("Vi tri 14")
                                    response_bot_final = response_bot
                                    intent_final = intent1
                                    confidence_recommendation = recommendation_score
                                    status = "recommend"
                                else:
                                    response_bot = get_answer_based_template(answer_database,
                                                                             "đề_xuất_khi_không_xác_định|đề_xuất") + "\n"
                                    response_bot += "\t+ " + suggestions[0] + "\n"
                                    recommendation_response.append(suggestions[0])
                                    response_bot = response_bot[:-1]
                                    # print("Vi tri 15")
                                    response_bot_final = response_bot
                                    intent_final = intent1
                                    confidence_recommendation = recommendation_score
                                    status = "recommend"
                            else:
                                
                                if conversation.context_error >= THRESHOLD_ERROR_FOR_AGENT:
                                    conversation.release_context_error()
                                    response_bot_final = (answer_database,
                                                                                   'chuyển_hội_thoại_cho_agent|agent')
                                    confidence_answer = 0.0
                                    status = "unknown"
                                    intent_final = "chuyển_hội_thoại_cho_agent"
                                else:
                                    conversation.update_context_error()                                    
                                    response_bot_final = get_answer_based_template(answer_database,
                                                                                   'hỏi_lại_khi_không_hiểu|hỏi_lại')
                                    confidence_answer = THRESHOLD_RETURN_CONFIDENCE
                                    status = "askagain"
                                    intent_final = "hỏi_lại_khi_không_hiểu"


    conversation.update_current_intent(intent_final)
    if status.strip() == "askagain":
        conversation.update_context_error()
    else:
        conversation.release_context_error()


    if len(slot_filling_result) > 10:
        
        list1 = [int(item[-1]) - int(item[-3]) for item in slot_filling_result]
        zipped_lists = zip(list1, slot_filling_result, slot_filling_candidate)
        sorted_pairs = sorted(zipped_lists)
        list1, slot_filling_result, slot_filling_candidate = [list(tuple) for tuple in zip(*sorted_pairs)]

    # """
    # print("confidence_answer: ", confidence_answer)
    print("preprocessed_input: ", user_query_clean)
    # print("intent_final: ", intent_final)
    # print("confidence_score1: ", confidence_score1)
    # print("pattern_template: ", str(pattern_template))
    # print("question_quality: ", question_quality)
    # print("response: ", response_bot_final)
    # print("sent_time bot: ", time.time())
    # print("sentiment: ", sentiment)
    # print("status: ", status)
    # print("Product: ", conversation.current_product)
    # print("slot_filling_template: ",slot_filling_candidate)
    # print("slot_filling_fill: ",slot_filling_result)
    # print("recommendation_confident_score: ", confidence_recommendation)
    # print("recommendation_response: ", recommendation_response)
    #print("=" * 100)

    ### Process slotfilling
    slot_filling = []
    for i_slot in range(len(slot_filling_result)):
        slot_filling_fill_modified = slot_filling_result[i_slot].upper()
        slot_filling_fill_modified = "|".join(slot_filling_fill_modified.split("|")[:-2])
        slot_filling_candidate_modified = "|".join(slot_filling_candidate[i_slot].split("|")[1:])
        slot_filling.append({
            "candidate_template": slot_filling_candidate_modified,
            "value": slot_filling_fill_modified
        })
    if len(slot_filling) > 10:
        slot_filling = slot_filling[:10]
    print(slot_filling)
    # """
    if type(pattern_template) is not list:
        pattern_template = [pattern_template]

    if confidence_recommendation >= 1.0:
        confidence_recommendation = 0.992424212

    output_json = {
        "confident_score": str(confidence_answer),
        "intent": intent_final,
        "pattern_template": pattern_template,
        "question_quality": question_quality,
        "response": response_bot_final,
        "sent_time": str(int(time.time())),
        "sentiment": sentiment,
        "session_id": str(session_id),
        "status": status,
        "product": conversation.current_product,
        "slot_filling": slot_filling,
        "recommendation_response": recommendation_response,
        "recommendation_confident_score": str(confidence_recommendation)
    }

    output_json_test = {
        "confident_score": str(confidence_answer),
        "preprocessed_input": user_query_clean,
        "intent": intent_final,
        "pattern_template": pattern_template,
        "response": response_bot_final,
        "session_id": str(session_id),
        "product": conversation.current_product,
        "slot_filling": slot_filling,
        "recommendation_response": recommendation_response,
        "recommendation_confident_score": str(confidence_recommendation)
    }

    # Save xuống database conversation
    conversation.save_database()
    # end44 = time.time()
    # print("Time for 1 response: ", end44 - start44)
    print("past:", conversation.past_product)
    print("current: ",conversation.current_product)
    return intent_final, confidence_score1, response_bot_final
