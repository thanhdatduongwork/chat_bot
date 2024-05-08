#!/usr/bin/env python

from APIOptimizer import *
from laserembeddings import Laser

##########################################################################

# LOAD CONFIGN FOR CHATBOT CIMB
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

# Chat xam CSDL

answer_database_chatxam = read_answer_chatxam()
question_chat_xam, template_chatxam = read_question_chatxam(synonyms_dictionary)
question_chat_xam_matrix = laser_model.embed_sentences(question_chat_xam, lang="vi")


def send():

    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "User: " + msg + '\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 8))
        session_id = 54312
        user_query = msg
        _, _, response_bot_final = get_response_api(user_query,session_id,
                                 intent_model,vectorizer,list_label, max_length,
                                 answer_database, answer_template,
                                 description_matrix, description_clean, description,
                                 list_product, list_product_clean, product_name_matrix,
                                 question_chat_xam, template_chatxam, question_chat_xam_matrix,answer_database_chatxam,
                                 laser_model, private_knowledge_df, synonyms_dictionary,
                                 THRESHOLD_INTENT_CONFIDENCE_DEFAULT,
                                 THRESHOLD_INTENT_CONFIDENCE_TRACKING,
                                 THRESHOLD_INTENT_CONFIDENCE_GREETING,
                                 THRESHOLD_ERROR_FOR_AGENT,
                                 NUMBER_OF_RECOMMENDATION_RESULT, THRESHOLD_RETURN_CONFIDENCE,
                                 THRESHOLD_INTENT_FOR_NOT_TRACKING)
        ChatLog.insert(END, "VKU-Bot: " + response_bot_final + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


if __name__ == '__main__':
    base = Tk()
    base.title("VKU Chatbot")
    base.geometry("400x500")
    base.resizable(width=FALSE, height=FALSE)


    # Create Chat window
    ChatLog = Text(base, bd=0, bg="white", height="8", width="80", font="Arial", )

    ChatLog.config(state=DISABLED)

    # Bind scrollbar to Chat window
    scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
    ChatLog['yscrollcommand'] = scrollbar.set

    # Create Button to send message
    SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                        bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                        command=send)

    # Create the box to enter message
    EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
    # EntryBox.bind("<Return>", send)

    # Place all components on the screen
    scrollbar.place(x=376, y=6, height=386)
    ChatLog.place(x=6, y=6, height=386, width=370)
    EntryBox.place(x=128, y=401, height=90, width=265)
    SendButton.place(x=6, y=401, height=90)

    base.mainloop()