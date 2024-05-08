# import pandas as pd

# file_name= "Data/Data.xlsx"
# data = pd.read_excel(file_name, sheet_name="Question", skiprows=2)
# y = data["Pattern Template"].tolist()

# intent = ['hỏi_đáp_nghề_nghiệp','hỏi_đáp_điểm_chuẩn','hỏi_đáp_ngành','hỏi_đáp_xét_tuyển','hỏi_đáp_ktx','hỏi_đáp_vku'
#           ,'hỏi_đáp_xe_bus','thông_tin_chỉ_tiêu','hỏi_đáp_tổ_hợp','thông_tin_học_bổng']
# list_keywords = []
# for item in set(y):
#     label = item.split("|")[0].strip()
#     if label == intent:
#         for keyword in  item.split("|")[1:]:
#             if keyword not in list_keywords:
#                 list_keywords.append(keyword)

# for item in list_keywords:
#     print(item)
import pandas as pd

file_name = "Data/Data.xlsx"
data = pd.read_excel(file_name, sheet_name="Question", skiprows=2)
y = data["Pattern Template"].tolist()


intent = ['hỏi_đáp_nghề_nghiệp','hỏi_đáp_điểm_chuẩn','hỏi_đáp_ngành','hỏi_đáp_xét_tuyển','hỏi_đáp_ktx','hỏi_đáp_vku'
           ,'hỏi_đáp_xe_bus','thông_tin_chỉ_tiêu','hỏi_đáp_tổ_hợp','thông_tin_học_bổng']
for intent in intent:
    list_keywords = []
    for item in set(y):
        label = item.split("|")[0].strip()
        if label == intent:
            for keyword in item.split("|")[1:]:
                if keyword not in list_keywords:
                    list_keywords.append(keyword)

    print(f"Intent: {intent}")
    for item in list_keywords:
        print(item)

