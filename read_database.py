import pickle
import pandas as pd
import tensorflow as tf
from preprocessing import *
from add_accents import *
import numpy as np
# Read trained model from the path of folder # Đọc mô hình đã huấn luyện từ đường dẫn thư mục
def read_trained_model(path='pickle_folder/intent_model.h5'):
    intent_model = tf.keras.models.load_model(path)
    return intent_model
#phục hồi dấu
def read_accent_model(path_model = "./pickle_folder/bilstm_5gram_fulldata.h5", NGRAM=5):
    accent_restore = AccentRestore(path_model, NGRAM = NGRAM)
    print("done load accent model!")
    return accent_restore
# Read vectorize from the path of folder Đọc vectorize từ đường dẫn thư mục
def read_vectorize(path='pickle_folder/vectorizer.pickle'):

    with open(path, 'rb') as handle:
        vectorizer = pickle.load(handle)

    return vectorizer

# Read list label
def read_list_label(path='pickle_folder/list_label.pickle'):
    with open(path, 'rb') as handle:
        list_label = pickle.load(handle)
    return list_label


# Read max length
def read_max_length(path='pickle_folder/max_length.pickle'):
    with open(path, 'rb') as handle:
        max_length = pickle.load(handle)
    return max_length

#Read_data
def Data(synonyms_dictionary,file_name="Data/Data.xlsx"):
    data = pd.read_excel(file_name, sheet_name="Question", skiprows=2)
    X = data["Question"].tolist()
    y = data["Pattern Template"].tolist()
    item_index = np.where(np.array(y) == 'truy_vấn_ngoài_phạm_vi|phạm_vi')[0]
    sentences = [processing_train(X[i],synonyms_dictionary) for i in item_index]
    return sentences



# Read data chatxam
def read_answer_chatxam(file_name= "Data/Data_Funny.xlsx"):
    answer_raw = pd.read_excel(file_name, sheet_name="Answer", skiprows=2)

    answer_expand = answer_raw.set_index('Pattern Template')
    answer_expand["Pattern Template"] = answer_expand.index
    answer_expand = answer_expand[["Answer", "Pattern Template"]]

    database_raw = answer_expand.copy()
    database = database_raw.set_index('Pattern Template').T.to_dict()

     # Check again to update multiple answer
    for key, value in database.items():
        rows = database_raw[database_raw['Pattern Template'] == key]
        rows_answer = rows['Answer'].to_list()
        value["Answer"] = rows_answer
    return database

# read question chatxam

def read_question_chatxam(synonyms_dictionary, file_name = "Data/Data_Funny.xlsx"):
    data = pd.read_excel(file_name, sheet_name="Funny_chat", skiprows=2)
    X = data["Question"].tolist()
    corr_templates = data["Pattern Template"].tolist()
    questions = []
    for item in X:
        questions.append(processing(item, synonyms_dictionary))
    return questions, corr_templates


# Read answer database
def read_answer_database(path="Data/Data.xlsx"):
    answer_raw = pd.read_excel(path, sheet_name="Answer", skiprows=2)
    intent_raw = pd.read_excel(path, sheet_name="Intent", skiprows=2)

    answer_expand = answer_raw.set_index('Pattern Template').join(intent_raw.set_index('Pattern Template'),
                                                                  rsuffix='_right')
    answer_expand["Pattern Template"] = answer_expand.index
    answer_expand = answer_expand[["Answer", "Pattern Template", "Description", "Product"]]   
    
    database_raw = answer_expand.copy()
    database = database_raw.set_index('Pattern Template').T.to_dict()

    # Check again to update multiple answer
    for key, value in database.items():
        rows = database_raw[database_raw['Pattern Template'] == key]
        rows_answer = rows['Answer'].to_list()
        value["Answer"] = rows_answer
    answer_template_unique = [str(item) for item in database.keys()]

    try:
        database["hỏi_lại_khi_không_hiểu|hỏi_lại"]["Answer"]
    except:
        answer_template_unique.append("hỏi_lại_khi_không_hiểu|hỏi_lại")
        database["hỏi_lại_khi_không_hiểu|hỏi_lại"] = \
            {"Answer":["VKU Bot không hiểu nội dung câu hỏi của bạn. Mong bạn vui lòng nhập câu hỏi rõ ràng hơn"],
             'Description': 'hỏi lại',
             'Product': 'Khác'
             }

    try:
        database["truy_vấn_ngoài_phạm_vi|phạm_vi"]["Answer"]
    except:
        answer_template_unique.append("truy_vấn_ngoài_phạm_vi|phạm_vi")
        database["truy_vấn_ngoài_phạm_vi|phạm_vi"] = \
            {"Answer": [
                "Câu hỏi này ngoài phạm vi trả lời của VKU Bot ạ"],
             'Description': 'phạm vi',
             'Product': 'Khác'
             }
    try:
        database["lời_chào_kết_thúc|kết_thúc"]["Answer"]
    except:
        answer_template_unique.append("lời_chào_kết_thúc|kết_thúc")
        database["lời_chào_kết_thúc|kết_thúc"] = \
            {"Answer": [
                "Cảm ơn bạn đã sử dụng VKU Bot. Rất vui khi hỗ trợ cho bạn"],
                'Description': 'phạm vi',
                'Product': 'Khác'
            }
    try:
        database["hỏi_đáp_không_ý_nghĩa|hỏi_đáp"]["Answer"]
    except:
        answer_template_unique.append("hỏi_đáp_không_ý_nghĩa|hỏi_đáp")
        database["hỏi_đáp_không_ý_nghĩa|hỏi_đáp"] = \
            {"Answer": [
                "Bạn có cần VKU Bot hỗ trợ giải đáp vấn đề gì nữa không?"],
                'Description': 'hỏi đáp',
                'Product': 'Khác'
            }

    try:
        database["lời_chào_mở_đầu|mở_đầu"]["Answer"]
    except:
        answer_template_unique.append("lời_chào_mở_đầu|mở_đầu")
        database["lời_chào_mở_đầu|mở_đầu"] = \
            {"Answer": [
                "Xin chào, VKU bot có thể hỗ trợ gì cho bạn!"],
                'Description': 'mở đầu',
                'Product': 'Khác'
            }

    try:
        database["chuyển_hội_thoại_cho_agent|agent"]["Answer"]
    except:
        answer_template_unique.append("chuyển_hội_thoại_cho_agent|agent")
        database["chuyển_hội_thoại_cho_agent|agent"] = \
            {"Answer": [
                "Bạn có thể liên hệ trực tiếp thông tin tư vấn của VKU Tuyển sinh để được giải đáp trực tiếp"],
                'Description': 'agent',
                'Product': 'Khác'
            }

    try:
        database["đề_xuất_khi_không_xác_định|đề_xuất"]["Answer"]
    except:
        answer_template_unique.append("đề_xuất_khi_không_xác_định|đề_xuất")
        database["đề_xuất_khi_không_xác_định|đề_xuất"] = \
            {"Answer": [
                "Xin lỗi không thể trả lời câu hỏi do sự nhập nhằng trong lựa chọn câu trả lời nên chúng tôi đề xuất các câu trả lời như sau:"],
                'Description': 'đề xuất',
                'Product': 'Khác'
            }

    return database,answer_template_unique


# Combine share knowledge with private knowledge
def read_knowledge(path_private="Data/Data.xlsx",path_share="Data/Data.xlsx"):
    private_knowledge = pd.read_excel(path_private, sheet_name="Private Knowledge", skiprows=2).fillna("")
    """
    for keyword in private_knowledge["Keyword"]:
        try:
            synonym = share_knowledge[keyword]["Synonym"]
            private_knowledge.loc[private_knowledge["Keyword"]==keyword, "Synonym"] += synonym
        except:
            continue
    """
    return private_knowledge


# read description
#THEM CODE DOT 4
def read_description(answer_database,answer_template,synonyms_dictionary):
    description = []
    for template in answer_template:
        try:
            if str(answer_database[template]["Description"]).strip() != "":
                description.append(answer_database[template]["Description"])
        except:
            continue

    description_clean = [processing(des,synonyms_dictionary) for des in description]
    description_clean2 = [clean_text(item) for item in description]
    return description,description_clean,description_clean2

def read_product(answer_database,answer_template):
    list_product = []
    for template in answer_template:
        try:
            if str(answer_database[template]["Product"]).strip() != "" and str(answer_database[template]["Product"]).strip() != "Khác":
                list_product.append(answer_database[template]["Product"])
        except:
            continue
    return set(list_product)


# extract promise template from dataset
def extract_template_from_database(intent, answer_template):
    template_list = [i for i in answer_template if i.startswith(intent)]
    #print(template_list)
    return template_list

def read_share_knowledge(path_share="Data/Data.xlsx"):

    df_synonym = pd.read_excel(path_share, sheet_name="Share Knowledge", skiprows=2)
    keyword = df_synonym["Keyword"].values.tolist()
    synonyms = df_synonym["Synonym"].values.tolist()
    synonyms_dictionary = dict(zip(keyword, synonyms))
    return synonyms_dictionary


def read_data_in_domain(synonyms_dictionary,file_name="Data/Data.xlsx"):
    data = pd.read_excel(file_name, sheet_name="Question", skiprows=2)
    X = data["Question"].tolist()
    sentences = [processing_train(item,synonyms_dictionary) for item in X]

    return sentences

def read_data_out_of_domain(synonyms_dictionary,file_name= "Data/OutDomain_final.xlsx"):
    data = pd.read_excel(file_name, sheet_name="Full out of domain", skiprows=2)
    X = data["Question"].tolist()
    sentences = [processing_train(item, synonyms_dictionary) for item in X]
    return sentences

def read_question_out_domain_have_ans(synonyms_dictionary, file_name="Data/OutDomain_final.xlsx"):
    data = pd.read_excel(file_name,sheet_name="Have answer")
    X = data["Questions"].tolist()
    list_ans_out_domain = data["Answers"].tolist()
    sentences = [processing_train(item, synonyms_dictionary) for item in X]
    return sentences,list_ans_out_domain

def read_dict_intent_description(synonyms_dictionary, file_name="Data/Data.xlsx"):    
    data = pd.read_excel(file_name, sheet_name = "Intent", skiprows=2)
    data = data.dropna()
    list_intents = data["Intent"].unique()
    dict_description = dict()
    dict_description_preprocessing = dict()
    for intent in list_intents:
        dict_description[intent] = data[data["Intent"]==intent]["Description"].tolist()
        dict_description_preprocessing[intent] = [processing_train(item, synonyms_dictionary) for item in data[data["Intent"]==intent]["Description"].tolist()]
    return dict_description,dict_description_preprocessing

        
if __name__ == '__main__':
    
    query = "tôi là sinh viên ngành cntt"
