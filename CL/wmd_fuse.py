import numpy as np
import pickle


def cal_performance(ranks, top=10):
    m_r = sum(ranks) * 1.0 / len(ranks)
    h_10 = sum(ranks <= top) * 1.0 / len(ranks)
    mrr = (1. / ranks).sum() / len(ranks)
    return m_r, h_10, mrr

fr = open('data/en_fr_15k_V1/mapping/0_3/' + 'stru_score.pk','rb')
records = pickle.load(fr)

### plus textual

inf = open('result_fr.txt')

counter = 0
ranks = []
for line in inf:
    strs = line.strip().split(' ')
    name_embed = [] # less is better
    for ind in range(1, 101):
        if strs[ind]!= 'nan':
            name_embed.append(float(strs[ind]))
        else:
            name_embed.append(float(100))
    name_embed = np.array(name_embed)
    stru_embed = records[counter] # more is better
    combined = name_embed - stru_embed
    rank = combined.argsort()
    rank_index = np.where(rank == 0)[0][0] + 1
    if rank_index ==0:
        print('wtf')
    #print (rank_index)
    ranks.append(rank_index)
    counter += 1

print(len(ranks))
print(ranks)
ranks = np.array(ranks)
MR, H10, MRR = cal_performance(ranks, top=10)
_, H1, _ = cal_performance(ranks, top=1)

msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
print('\n'+msg)

