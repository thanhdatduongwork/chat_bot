from flask import Flask, request, render_template
from flask_cors import  CORS
from APIOptimizer import *
from laserembeddings import Laser
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["UPLOAD_FOLDER"] = "./static"


##########################################################################

# LOAD CONFIGN FOR CHATBOT CIMB
with open("./config/chatbot_config.txt", "r",encoding='utf-8') as file:
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

### load accent model
accent_restore = read_accent_model()

Out_intent_model = read_trained_model(path="pickle_folder/Out_intent_model.h5")
Out_vectorizer = read_vectorize(path="pickle_folder/Out_vectorizer.pickle")
Out_max_length = read_max_length(path="pickle_folder/Out_max_length.pickle")


answer_database, answer_template = read_answer_database()
private_knowledge_df = read_knowledge(path_private="Data/Data.xlsx",
                                      path_share="Data/Data.xlsx")

# BO SUNG THEM DOT 3
synonyms_dictionary = read_share_knowledge(path_share="Data/Data.xlsx")
description,description_clean,description_clean2 = read_description(answer_database, answer_template, synonyms_dictionary)

# SAI Ở ĐÂY
description_matrix =  laser_model.embed_sentences(description_clean, lang='vi')
list_product = read_product(answer_database,answer_template)
#THEM
list_product_clean = [processing(product_name,synonyms_dictionary) for product_name in list_product]
#THEM
product_name_matrix = laser_model.embed_sentences(list_product_clean, lang='vi')



answer_database_chatxam = read_answer_chatxam()
question_chat_xam, template_chatxam = read_question_chatxam(synonyms_dictionary)
question_chat_xam_matrix = laser_model.embed_sentences(question_chat_xam, lang="vi")

question_out_domain, list_ans_out_domain = read_question_out_domain_have_ans(synonyms_dictionary)
question_out_domain_matrix = laser_model.embed_sentences(question_out_domain, lang="vi")


### get dict questions by intent

dict_description,dict_description_preprocessing = read_dict_intent_description(synonyms_dictionary)
dict_description_matrix = dict()
for intent in dict_description:
    print("="*20)
    print(intent)
    print(len(dict_description[intent]))
    dict_description_matrix[intent] = laser_model.embed_sentences(dict_description[intent], lang="vi")
@app.route('/')
# flask routing
@app.route('/api/get_answer', methods=["POST", "GET"])
def get_answer():
    try:
        question = request.form.get("question")
        session_id = request.form.get("session")
    except:
        question = request.args.get('question')
        session_id = request.form.get("session")

    print(session_id)
    user_query = question
    intent_final, _, response_bot_final = get_response_api(user_query, session_id,
                                                intent_model, vectorizer, list_label, max_length,
                                                Out_intent_model, Out_vectorizer, Out_max_length,
                                                answer_database, answer_template,
                                                description_matrix, description_clean, description,
                                                list_product, list_product_clean, product_name_matrix,
                                                question_chat_xam, template_chatxam, question_chat_xam_matrix,
                                                answer_database_chatxam,
                                                laser_model, private_knowledge_df, synonyms_dictionary,accent_restore,
                                                question_out_domain_matrix, list_ans_out_domain,
                                                dict_description,dict_description_preprocessing,dict_description_matrix,
                                                THRESHOLD_INTENT_CONFIDENCE_DEFAULT,
                                                THRESHOLD_INTENT_CONFIDENCE_TRACKING,
                                                THRESHOLD_INTENT_CONFIDENCE_GREETING,
                                                THRESHOLD_ERROR_FOR_AGENT,
                                                NUMBER_OF_RECOMMENDATION_RESULT, THRESHOLD_RETURN_CONFIDENCE,
                                                THRESHOLD_INTENT_FOR_NOT_TRACKING)
    print()
    print("="*20)
    print(intent_final)
    return  {
            "code" : 0,
            "answer" : response_bot_final 
            }   

if __name__ == '__main__':    
    app.run(host = '0.0.0.0', port = 8080, debug=False)

