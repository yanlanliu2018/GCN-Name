import numpy as np
import pickle
import re
#import jieba
##load discriptional text
id2name = dict()

def rm_comma(text):
    return re.sub("[\s+\.\!\/_,$%^*(·)-+\"\']+|[+——！！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.，。？、~@#￥%：……&*（）]+", " ",text)

#path = 'data/dbp_wd_15k_V1/mapping/0_3/'
path = 'data/en_de_15k_V1/mapping/0_3/'

disinput1 = open(path + 'ent_ids_1')
for line in disinput1:
    strs = line.strip().split('\t')
    id2name[int(strs[0])] = rm_comma(strs[1].split('/')[-1].lower())
    #words = rm_comma(strs[1].split('/')[-1].lower())
    #seg_list = jieba.cut(words, cut_all=False)
    #strr = " ".join(seg_list)
    #id2name[int(strs[0])] = strr
print(len(id2name))

disinput2 = open(path + 'ent_ids_2')
for line in disinput2:
    strs = line.strip().split('\t')
    id2name[int(strs[0])] = rm_comma(strs[1].split('/')[-1].lower())
print(len(id2name))


###
fr = open('en_de_15k_V1/' + 'records_l2r.pk','rb')
records = pickle.load(fr)
print(len(records))
counter = 0

str_scores_all = []
for record in records:
    #print(record)
    # if counter > 3: break
    # counter += 1
    [trueid, tragetid, ids, scores, str_scores] = record
    #print(record)
    ids = ids.tolist()
    # #print(ids)
    if trueid in id2name and tragetid in id2name and tragetid in ids:
        #print(counter)
        truedis = id2name[trueid]
        targetdis = id2name[tragetid]
        outf = open('de_trainning/'+str(counter)+'.txt', 'w')
        counter += 1
        outf.write(str(trueid) + '\t' + truedis +'\n')
        outf.write(str(tragetid) + '\t' + targetdis +'\n')
        outf.flush()
        ids.remove(tragetid)
        for id in ids:
            if id in id2name:
                dis = id2name[id]
            else:
                dis = ' '
            outf.write(str(id) + '\t' + dis +'\n')
            outf.flush()
        str_scores_all.append(str_scores)

df2=open(path + 'stru_score.pk','wb')
pickle.dump(str_scores_all, df2)
