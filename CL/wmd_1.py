import numpy as np
import scipy
from scipy import spatial
import pickle
import tensorflow as tf

def cal_performance(ranks, top=10):
    m_r = sum(ranks) * 1.0 / len(ranks)
    h_10 = sum(ranks <= top) * 1.0 / len(ranks)
    mrr = (1. / ranks).sum() / len(ranks)
    return m_r, h_10, mrr

def get_combine_hits_select_correct(vec, name_vec, test_pair, dicrank, max_correct):
	records = []
	Lvec = tf.placeholder(tf.float32, [None, 300])
	Rvec = tf.placeholder(tf.float32, [None, 300])
	he = tf.nn.l2_normalize(Lvec, dim=-1) #??? 规范化啊
	norm_e_em = tf.nn.l2_normalize(Rvec, dim=-1)
	aep = tf.matmul(he, tf.transpose(norm_e_em))

	Lvec_ne = tf.placeholder(tf.float32, [None, 900])
	Rvec_ne = tf.placeholder(tf.float32, [None, 900])
	he_n = tf.nn.l2_normalize(Lvec_ne, dim=-1) #??? 规范化啊
	norm_e_em_n = tf.nn.l2_normalize(Rvec_ne, dim=-1)
	aep_n = tf.matmul(he_n, tf.transpose(norm_e_em_n))

	sess = tf.Session()
	Lv = np.array([vec[e1] for e1, e2 in test_pair])
	Lid_record = np.array([e1 for e1, e2 in test_pair])
	Rv = np.array([vec[e2] for e1, e2 in test_pair])
	Rid_record = np.array([e2 for e1, e2 in test_pair])

	Lv_ne = np.array([name_vec[e1] for e1, e2 in test_pair])
	Rv_ne = np.array([name_vec[e2] for e1, e2 in test_pair])
	aep = sess.run(aep, feed_dict = {Lvec: Lv, Rvec: Rv})
	aep_n = sess.run(aep_n, feed_dict = {Lvec_ne: Lv_ne, Rvec_ne: Rv_ne})
	aep_fuse = aep*0.3+ aep_n*0.7
	aep_fuse_r = aep_fuse.T

	probs = aep_fuse - aep_fuse[range(len(Lid_record)), range(len(Lid_record))].reshape(len(aep_fuse), 1)
	# only rank those who have correspondings... cause the set is distorted for those above max_correct
	ranks = (probs >= 0).sum(axis=1)[:max_correct]
	cannotmactch = [10000]* (len(test_pair) - max_correct) #### 好烦啊，这个其实不知道他已经没有对象了，但是为了方便计算才从全局角度知道没有对象，每一顿还得加入训练。。。
	cannotmactch = np.array(cannotmactch)
	ranks = np.append(ranks, cannotmactch)

	truth =  np.where(ranks==1)
	truths = truth[0].tolist()
	ind = np.argmax(probs, axis= 1)[:max_correct]
	#ind = np.append(ind, np.array(cannotmactch))

	# probs is a matrix... now need the rank within each row (!!!)
	for i in range(len(test_pair)): #i就对应是几号
		struc_scores = aep[i, :]
		scores = probs[i, :]
		rank_score = np.sort(scores)
		rank_score = np.flip(rank_score, axis = -1)[:100] # 从大到小的前几个。。。 这个是相似度，另一个是距离。。。

		rank = (-scores).argsort() #从大到小
		#if rank[0] != i: #这一轮不正确
		rank = rank[:100]

		struc_scores = struc_scores[rank] # record the structural scores of these entities!!!

		ranked_nodes = Rid_record[rank]
		target_node = Lid_record[i]
		true_node = Rid_record[i]
		records.append([target_node, true_node, ranked_nodes, rank_score, struc_scores])

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap = maxes-maxes1
	gap = gap[:max_correct]

	## pre
	pre = []
	for ent in dicrank.keys():
		pre.append(dicrank[ent])
	pre = np.array(pre)
	#ranks_new = np.append(ranks, pre)
	ranks_new = ranks
	print('to be evaluated... ' + str(len(ranks_new)))

	MR, H10, MRR = cal_performance(ranks_new, top=10)
	_, H1, _ = cal_performance(ranks_new, top=1)

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print(msg)

	print('pre evaluated... ' + str(len(pre)))

	MR, H10, MRR = cal_performance(pre, top=10)
	_, H1, _ = cal_performance(pre, top=1)

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print(msg)


	return records


#path =  'en_de_15k_V1/'
path =  'dbp_wd_15k_V1/'
itea = 12
test = np.load(path + 'test' + str(itea)+ '.npy')
se_vec = np.load(path + 'se_vec' + str(itea)+ '.npy')
ne_vec = np.load(path + 'ne_vec.npy')
fr = open(path + 'dicrank' + str(itea)+ '.npy','rb')


dicrank = pickle.load(fr)
# need max in test...
records = get_combine_hits_select_correct(se_vec, ne_vec, test, dicrank, len(test))
print(len(records))
df2=open(path + 'records_l2r.pk','wb')
pickle.dump(records,df2)
print('len of test ' + str(len(test)))


# decide those who have been added to seeds will not be re-ranked!
# only rank the rest and those have no corresponds would also not be ranked!!!



# df2=open('/home/weixin/Desktop/zwx/en_fr/base+t+nv+ite1/records_r2l.pk','wb')
# pickle.dump(records,df2)
