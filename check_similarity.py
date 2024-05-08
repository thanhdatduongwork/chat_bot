import pandas as pd
import re

def extract_ngram(sentence, N):
    sentence = sentence.split(" ")
    grams = [sentence[i:i+N] for i in range(len(sentence)-N+1)]
    return grams


def Levenstein_distance(s0, s1):
    if s0 is None:
        raise TypeError("Argument s0 is NoneType.")
    if s1 is None:
        raise TypeError("Argument s1 is NoneType.")
    if s0 == s1:
        return 0.0
    if len(s0) == 0:
        return len(s1)
    if len(s1) == 0:
        return len(s0)

    v0 = [0] * (len(s1) + 1)
    v1 = [0] * (len(s1) + 1)

    for i in range(len(v0)):
        v0[i] = i

    for i in range(len(s0)):
        v1[0] = i + 1
        for j in range(len(s1)):
            cost = 1
            if s0[i] == s1[j]:
                cost = 0
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0

    return v0[len(s1)]


# Combine share knowledge with private knowledge
def read_knowledge(path_private, path_share):
    share_knowledge =  pd.read_excel(path_share, sheet_name="Share Knowledge", skiprows=2).set_index('Keyword').T.to_dict()
    private_knowledge = pd.read_excel(path_private, sheet_name="Private Knowledge", skiprows=2).fillna("")
    for index, row in private_knowledge.iterrows():
        keywords_list = row["Synonym"].split(",")
        keyword = row["Keyword"]
        try:
            synonym = share_knowledge[keyword]["Synonym"].split(",")
            for item in synonym:
                if item not in keywords_list:
                    keywords_list.append(item.strip())
        except:
            continue

        keyword2 = row["Synonym"].split(",")
        for item2 in keyword2:
            try:
                synonym3 = share_knowledge[item2]["Synonym"].split(",")
                for item3 in synonym3:
                    if item3 not in keywords_list:
                        keywords_list.append(item3.strip())
            except:
                continue
        s = ""
        for item in keywords_list:
            if item.strip() != "":
                s += item.strip() + ","
        s = s[:-1]
        private_knowledge.loc[index, "Synonym"] = s

    return private_knowledge


# Check similarity based on Levenstein approach
def check_similarity_Levenstein(entity, sentence):
    n = len(entity.split(" "))
    grams = extract_ngram(sentence, n)

    for grams in grams:
        grams = " ".join(grams)
        distance = Levenstein_distance(grams, entity)
        if distance < 2:
            return True
    return False

# clean text to check
def clean_text(text):
    text = str(text).replace('_', " ").lower().strip()
    return text
# Using synonym word of entity to check
def check_synonym(keyword, intent ,user_query, private_knowledge_df):
    try:
        list_synonym = private_knowledge_df[(private_knowledge_df["Keyword"].str.lower() == keyword) &
          (private_knowledge_df['Intent'] == intent)]["Synonym"].tolist()[0]
        for synonym in list_synonym.split(","):
            key = clean_text(synonym)
            if key != "":
                sent = clean_text(user_query)
                if ".*" not in key:                
                    if re.search(r'\b' + key + r'\b', sent) != None or (len(key.split(" ")) > 1 and check_similarity_Levenstein(key,sent) == True):
                        return True
                else: 
                    r1 = re.findall(key,sent)
                    if len(r1) > 0:
                        return True
        return False
    except:
        print("check_synonym error")
        return False


if __name__ == '__main__':
    sent = "biết giá phòng 2 sinh viên"
    key = "phòng 4"
    print(check_similarity_Levenstein(key, sent))

    print(re.search(r'\b' + key + r'\b', sent))