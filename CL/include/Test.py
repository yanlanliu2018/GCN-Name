import numpy as np
import scipy
from scipy.spatial.distance import cdist
import tensorflow as tf
import copy

def cal_performance(ranks, top=10):
    m_r = sum(ranks) * 1.0 / len(ranks)
    h_10 = sum(ranks <= top) * 1.0 / len(ranks)
    mrr = (1. / ranks).sum() / len(ranks)
    return m_r, h_10, mrr

def get_hits(vec, test_pair, outfile, top_k=(1, 10, 50, 100)):
	Lvec = np.array([vec[e1] for e1, e2 in test_pair])
	Rvec = np.array([vec[e2] for e1, e2 in test_pair])
	sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
	top_lr = [0] * len(top_k)
	mrr_sum_l = 0
	for i in range(Lvec.shape[0]):
		rank = sim[i, :].argsort()
		rank_index = np.where(rank == i)[0][0]
		mrr_sum_l = mrr_sum_l + 1.0/(rank_index+1)
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_lr[j] += 1
	top_rl = [0] * len(top_k)
	mrr_sum_r = 0
	for i in range(Rvec.shape[0]):
		rank = sim[:, i].argsort()
		rank_index = np.where(rank == i)[0][0]
		mrr_sum_r = mrr_sum_r + 1.0/(rank_index+1)
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_rl[j] += 1
	print('For each left:')
	outfile.write('For each left:\n')
	for i in range(len(top_lr)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
		outfile.write('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100) + '\n')
	print("MRR: " + str(mrr_sum_l/len(test_pair)))
	outfile.write("MRR: " + str(mrr_sum_l/len(test_pair)) + '\n')
	print('For each right:')
	outfile.write('For each right:\n')
	for i in range(len(top_rl)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
		outfile.write('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100) + '\n')
	outfile.write("MRR: " + str(mrr_sum_r/len(test_pair)) + '\n')
	print("MRR: " + str(mrr_sum_r/len(test_pair)))
	outfile.flush()

def get_combine_hits_select(se_vec, ne_vec, beta, test_pair, outfile, theta_3, top_k=(1, 10, 50, 100)):
	Lvec_se = np.array([se_vec[e1] for e1, e2 in test_pair])
	Lid_record = np.array([e1 for e1, e2 in test_pair])
	Rvec_se = np.array([se_vec[e2] for e1, e2 in test_pair])
	Rid_record = np.array([e2 for e1, e2 in test_pair])
	sim_se = scipy.spatial.distance.cdist(Lvec_se, Rvec_se, metric='cityblock')
	Lvec_ne = np.array([ne_vec[e1] for e1, e2 in test_pair])
	Rvec_ne = np.array([ne_vec[e2] for e1, e2 in test_pair])
	sim_ne = scipy.spatial.distance.cdist(Lvec_ne, Rvec_ne, metric='cityblock')
	LL = len(test_pair)
	top_lr = [0] * len(top_k)
	mrr_sum_l = 0
	left2right = dict()
	for i in range(LL):
		sim = sim_se[i, :] * beta + sim_ne[i, :] * (1.0 - beta)
		rank = sim.argsort()
		lid = Lid_record[i]
		rid = Rid_record[rank[0]]
		left2right[lid] = rid
		rank_index = np.where(rank == i)[0][0]
		mrr_sum_l = mrr_sum_l + 1.0/(rank_index+1)
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_lr[j] += 1
	top_rl = [0] * len(top_k)
	mrr_sum_r = 0
	right2left = dict()
	for i in range(LL):
		sim = sim_se[:, i] * beta + sim_ne[:, i] * (1.0 - beta)
		rank = sim.argsort()
		rid = Rid_record[i]
		lid = Lid_record[rank[0]]
		right2left[rid] = lid
		rank_index = np.where(rank == i)[0][0]
		mrr_sum_r = mrr_sum_r + 1.0/(rank_index+1)
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_rl[j] += 1
	print('For each left:')
	outfile.write('For each left:\n')
	for i in range(len(top_lr)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / LL * 100))
		outfile.write('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / LL * 100) + '\n')
	outfile.write("MRR: " + str(mrr_sum_l/len(test_pair)) + '\n')
	print('For each right:')
	outfile.write('For each right:\n')
	for i in range(len(top_rl)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / LL * 100))
		outfile.write('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / LL * 100) + '\n')
	outfile.write("MRR: " + str(mrr_sum_r/len(test_pair)) + '\n')

	matched = []
	for item in left2right:
		if left2right[item] not in right2left:
			print('error')
		else:
			if right2left[left2right[item]] == item:
				# item_embed_se = np.array(se_vec[item])
				# item_embed_se1 = np.array(se_vec[left2right[item]])
				# dis_se = scipy.spatial.distance.cityblock(item_embed_se, item_embed_se1)
				# item_embed_ne = np.array(ne_vec[item])
				# item_embed_ne1 = np.array(ne_vec[left2right[item]])
				# dis_ne = scipy.spatial.distance.cityblock(item_embed_ne, item_embed_ne1)
				# dis = dis_se * beta + dis_ne * (1.0 - beta)
				# if dis <= theta_3:
					matched.append((item, left2right[item]))
	print(len(matched))
	outfile.write(str(len(matched)) + '\n\n')
	outfile.flush()
	print()

	return matched

def get_combine_hits_select_rough(se_vec, ne_vec, beta, test_pair, outfile, softdegree, theta_4, top_k=(1, 10, 50, 100)):
	Lvec_se = np.array([se_vec[e1] for e1, e2 in test_pair])
	Lid_record = np.array([e1 for e1, e2 in test_pair])
	Rvec_se = np.array([se_vec[e2] for e1, e2 in test_pair])
	Rid_record = np.array([e2 for e1, e2 in test_pair])
	sim_se = scipy.spatial.distance.cdist(Lvec_se, Rvec_se, metric='cityblock')
	Lvec_ne = np.array([ne_vec[e1] for e1, e2 in test_pair])
	Rvec_ne = np.array([ne_vec[e2] for e1, e2 in test_pair])
	sim_ne = scipy.spatial.distance.cdist(Lvec_ne, Rvec_ne, metric='cityblock')
	LL = len(test_pair)
	top_lr = [0] * len(top_k)
	left2right = dict()
	mrr_sum_l = 0
	for i in range(LL):
		sim = sim_se[i, :] * beta + sim_ne[i, :] * (1.0 - beta)
		rank = sim.argsort()
		lid = Lid_record[i]
		rid = Rid_record[rank[0:softdegree]]
		left2right[lid] = rid
		rank_index = np.where(rank == i)[0][0]
		mrr_sum_l = mrr_sum_l + 1.0/(rank_index+1)
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_lr[j] += 1
	top_rl = [0] * len(top_k)
	right2left = dict()
	mrr_sum_r = 0
	for i in range(LL):
		sim = sim_se[:, i] * beta + sim_ne[:, i] * (1.0 - beta)
		rank = sim.argsort()
		rid = Rid_record[i]
		lid = Lid_record[rank[0:softdegree]]
		right2left[rid] = lid
		rank_index = np.where(rank == i)[0][0]
		mrr_sum_r = mrr_sum_r + 1.0/(rank_index+1)
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_rl[j] += 1
	print('For each left:')
	outfile.write('For each left:\n')
	for i in range(len(top_lr)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / LL * 100))
		outfile.write('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / LL * 100) + '\n')
	outfile.write("MRR: " + str(mrr_sum_l/len(test_pair)) + '\n')
	print('For each right:')
	outfile.write('For each right:\n')
	for i in range(len(top_rl)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / LL * 100))
		outfile.write('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / LL * 100) + '\n')
	outfile.write("MRR: " + str(mrr_sum_r/len(test_pair)) + '\n')

	matched = []
	for item in left2right:
		# item [1,2,3,4,5]
		if left2right[item][0] not in right2left:
			print('error')
		else:
			item_embed_se = np.array(se_vec[item])
			item_embed_ne = np.array(ne_vec[item])
			if right2left[left2right[item][0]][0] == item:
				# item_embed_se1 = np.array(se_vec[left2right[item][0]])
				# dis_se = scipy.spatial.distance.cityblock(item_embed_se, item_embed_se1)
				# item_embed_ne1 = np.array(ne_vec[left2right[item][0]])
				# dis_ne = scipy.spatial.distance.cityblock(item_embed_ne, item_embed_ne1)
				# dis = dis_se * beta + dis_ne * (1.0 - beta)
				# if dis <= theta_4:
					matched.append((item, left2right[item][0]))
			else:
				for it in left2right[item]:
					if item in right2left[it]:
						# item_embed_se1 = np.array(se_vec[it])
						# dis_se = scipy.spatial.distance.cityblock(item_embed_se, item_embed_se1)
						# item_embed_ne1 = np.array(ne_vec[it])
						# dis_ne = scipy.spatial.distance.cityblock(item_embed_ne, item_embed_ne1)
						# dis = dis_se * beta + dis_ne * (1.0 - beta)
						# if dis <= theta_4:
							matched.append((item, it))
							continue
	print(len(matched))
	outfile.write(str(len(matched)) + '\n\n')
	outfile.flush()
	print()

	return matched

def get_combine_hits_select_ct(se_vec, ne_vec, beta, test_pair, outfile, id2fre, theta_3, top_k=(1, 10, 50, 100)):
	Lvec_se = np.array([se_vec[e1] for e1, e2 in test_pair])
	Lid_record = np.array([e1 for e1, e2 in test_pair])
	Rvec_se = np.array([se_vec[e2] for e1, e2 in test_pair])
	Rid_record = np.array([e2 for e1, e2 in test_pair])
	sim_se = cdist(Lvec_se, Rvec_se, metric='cityblock')
	Lvec_ne = np.array([ne_vec[e1] for e1, e2 in test_pair])
	Rvec_ne = np.array([ne_vec[e2] for e1, e2 in test_pair])
	sim_ne = cdist(Lvec_ne, Rvec_ne, metric='cityblock')
	LL = len(test_pair)
	top_lr = [0] * len(top_k)
	mrr_sum_l = 0
	left2right = dict()
	for i in range(LL):
		sim = sim_se[i, :] * beta + sim_ne[i, :] * (1.0 - beta)
		rank = sim.argsort()
		lid = Lid_record[i]
		rid = Rid_record[rank[0]]
		left2right[lid] = rid
		rank_index = np.where(rank == i)[0][0]
		mrr_sum_l = mrr_sum_l + 1.0/(rank_index+1)
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_lr[j] += 1
	top_rl = [0] * len(top_k)
	mrr_sum_r = 0
	right2left = dict()
	for i in range(LL):
		sim = sim_se[:, i] * beta + sim_ne[:, i] * (1.0 - beta)
		rank = sim.argsort()
		rid = Rid_record[i]
		lid = Lid_record[rank[0]]
		right2left[rid] = lid
		rank_index = np.where(rank == i)[0][0]
		mrr_sum_r = mrr_sum_r + 1.0/(rank_index+1)
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_rl[j] += 1
	print('For each left:')
	outfile.write('For each left:\n')
	for i in range(len(top_lr)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / LL * 100))
		outfile.write('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / LL * 100) + '\n')
	outfile.write("MRR: " + str(mrr_sum_l/len(test_pair)) + '\n')
	print('For each right:')
	outfile.write('For each right:\n')
	for i in range(len(top_rl)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / LL * 100))
		outfile.write('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / LL * 100) + '\n')
	outfile.write("MRR: " + str(mrr_sum_r/len(test_pair)) + '\n')

	matched = []
	for item in left2right:
		if left2right[item] not in right2left:
			print('error')
		else:
			if right2left[left2right[item]] == item:
				if id2fre[str(item)] >= theta_3:
				# item_embed_se = np.array(se_vec[item])
				# item_embed_se1 = np.array(se_vec[left2right[item]])
				# dis_se = scipy.spatial.distance.cityblock(item_embed_se, item_embed_se1)
				# item_embed_ne = np.array(ne_vec[item])
				# item_embed_ne1 = np.array(ne_vec[left2right[item]])
				# dis_ne = scipy.spatial.distance.cityblock(item_embed_ne, item_embed_ne1)
				# dis = dis_se * beta + dis_ne * (1.0 - beta)
				# print(dis)
				# if dis <= theta_3:
					matched.append((item, left2right[item]))
	print(len(matched))
	outfile.write(str(len(matched)) + '\n\n')
	outfile.flush()
	print()

	return matched

def get_hits_select(vec, test_pair, outfile, top_k=(1, 10, 50, 100)):
	Lvec = np.array([vec[e1] for e1, e2 in test_pair])
	Lid_record = np.array([e1 for e1, e2 in test_pair])
	Rvec = np.array([vec[e2] for e1, e2 in test_pair])
	Rid_record = np.array([e2 for e1, e2 in test_pair])
	sim = cdist(Lvec, Rvec, metric='cityblock')
	top_lr = [0] * len(top_k)
	mrr_sum_l = 0
	left2right = dict()
	for i in range(Lvec.shape[0]):
		rank = sim[i, :].argsort()
		lid = Lid_record[i]
		rid = Rid_record[rank[0]]
		left2right[lid] = rid
		rank_index = np.where(rank == i)[0][0]
		mrr_sum_l = mrr_sum_l + 1.0/(rank_index+1)
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_lr[j] += 1
	mrr_sum_r = 0
	top_rl = [0] * len(top_k)
	right2left = dict()
	for i in range(Rvec.shape[0]):
		rank = sim[:, i].argsort()
		rid = Rid_record[i]
		lid = Lid_record[rank[0]]
		right2left[rid] = lid
		rank_index = np.where(rank == i)[0][0]
		mrr_sum_r = mrr_sum_r + 1.0/(rank_index+1)
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_rl[j] += 1
	print('For each left:')
	outfile.write('For each left:\n')
	for i in range(len(top_lr)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
		outfile.write('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100) + '\n')
	outfile.write("MRR: " + str(mrr_sum_l/len(test_pair)) + '\n')
	print("MRR: " + str(mrr_sum_l/len(test_pair)))
	print('For each right:')
	outfile.write('For each right:\n')
	for i in range(len(top_rl)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
		outfile.write('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100) + '\n')
	outfile.write("MRR: " + str(mrr_sum_r/len(test_pair)) + '\n')
	print("MRR: " + str(mrr_sum_r/len(test_pair)))
	outfile.flush()
	matched = []
	for item in left2right:
		if left2right[item] not in right2left:
			print('wtf')
		else:
			if right2left[left2right[item]] == item:
				matched.append((item, left2right[item]))
	#print(matched)
	print(len(matched))
	outfile.write(str(len(matched)) + '\n\n')
	outfile.flush()
	print()

	return matched

def get_hits_select_correct(vec, test_pair, dicrank, max_correct):
	Lvec = tf.placeholder(tf.float32, [None, 300])
	Rvec = tf.placeholder(tf.float32, [None, 300])

	he = tf.nn.l2_normalize(Lvec, dim=-1) #??? 规范化啊
	norm_e_em = tf.nn.l2_normalize(Rvec, dim=-1)
	aep = tf.matmul(he, tf.transpose(norm_e_em))

	sess = tf.Session()
	Lv = np.array([vec[e1] for e1, e2 in test_pair])
	Lid_record = np.array([e1 for e1, e2 in test_pair])
	Rv = np.array([vec[e2] for e1, e2 in test_pair])
	Rid_record = np.array([e2 for e1, e2 in test_pair])
	aep_fuse = sess.run(aep, feed_dict = {Lvec: Lv, Rvec: Rv})
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

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap = maxes-maxes1
	gap = gap[:max_correct]

	### pre
	pre = []
	for ent in dicrank.keys():
		pre.append(dicrank[ent])
	pre = np.array(pre)
	ranks_new = np.append(ranks, pre)
	print('to be evaluated... ' + str(len(ranks_new)))

	MR, H10, MRR = cal_performance(ranks_new, top=10)
	_, H1, _ = cal_performance(ranks_new, top=1)

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	# right
	probs = aep_fuse_r - aep_fuse_r[range(len(Lid_record)), range(len(Lid_record))].reshape(len(aep_fuse_r), 1)
	ranks_r = (probs >= 0).sum(axis=1)[:max_correct]
	truth_r =  np.where(ranks_r==1)
	truths_r = truth_r[0].tolist()
	ind_r = np.argmax(probs, axis= 1)[:max_correct]

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap_r = maxes-maxes1
	gap_r = gap_r[:max_correct]

	ranks_r = np.append(ranks_r, cannotmactch)

	MR, H10, MRR = cal_performance(ranks_r, top=10)
	_, H1, _ = cal_performance(ranks_r, top=1)

	print('to be evaluated... ' + str(len(ranks_r)))

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	return ind, gap, truths, ranks,  ind_r, gap_r, truths_r, ranks_r

def get_combine_hits_select_correct(vec, name_vec, test_pair, dicrank, max_correct):
	Lvec = tf.placeholder(tf.float32, [None, 900])
	Rvec = tf.placeholder(tf.float32, [None, 900])
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
	alpha = 0.8
	aep_fuse = aep*alpha+ aep_n*(1-alpha)
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

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap = maxes-maxes1
	gap = gap[:max_correct]

	### pre
	pre = []
	for ent in dicrank.keys():
		pre.append(dicrank[ent])
	pre = np.array(pre)
	ranks_new = np.append(ranks, pre)
	print('to be evaluated... ' + str(len(ranks_new)))

	MR, H10, MRR = cal_performance(ranks_new, top=10)
	_, H1, _ = cal_performance(ranks_new, top=1)

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	# right
	probs = aep_fuse_r - aep_fuse_r[range(len(Lid_record)), range(len(Lid_record))].reshape(len(aep_fuse_r), 1)
	ranks_r = (probs >= 0).sum(axis=1)[:max_correct]
	truth_r =  np.where(ranks_r==1)
	truths_r = truth_r[0].tolist()
	ind_r = np.argmax(probs, axis= 1)[:max_correct]

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap_r = maxes-maxes1
	gap_r = gap_r[:max_correct]

	ranks_r = np.append(ranks_r, cannotmactch)

	MR, H10, MRR = cal_performance(ranks_r, top=10)
	_, H1, _ = cal_performance(ranks_r, top=1)

	print('to be evaluated... ' + str(len(ranks_r)))

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	return ind, gap, truths, ranks,  ind_r, gap_r, truths_r, ranks_r

def get_hits_select_correct_1(vec, test_pair):
	Lvec = tf.placeholder(tf.float32, [None, 300])
	Rvec = tf.placeholder(tf.float32, [None, 300])

	he = tf.nn.l2_normalize(Lvec, dim=-1) #??? 规范化啊
	norm_e_em = tf.nn.l2_normalize(Rvec, dim=-1)
	aep = tf.matmul(he, tf.transpose(norm_e_em))

	sess = tf.Session()
	Lv = np.array([vec[e1] for e1, e2 in test_pair])
	Lid_record = np.array([e1 for e1, e2 in test_pair])
	Rv = np.array([vec[e2] for e1, e2 in test_pair])
	Rid_record = np.array([e2 for e1, e2 in test_pair])
	aep_fuse = sess.run(aep, feed_dict = {Lvec: Lv, Rvec: Rv})
	aep_fuse_r = aep_fuse.T

	probs = aep_fuse - aep_fuse[range(len(Lid_record)), range(len(Lid_record))].reshape(len(aep_fuse), 1)
	# only rank those who have correspondings... cause the set is distorted for those above max_correct
	ranks = (probs >= 0).sum(axis=1)
	truth =  np.where(ranks==1)
	truths = truth[0].tolist()
	ind = np.argmax(probs, axis= 1)
	#ind = np.append(ind, np.array(cannotmactch))

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap = maxes-maxes1
	gap = gap

	### pre
	print('to be evaluated... ' + str(len(ranks)))

	MR, H10, MRR = cal_performance(ranks, top=10)
	_, H1, _ = cal_performance(ranks, top=1)

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	# right
	probs = aep_fuse_r - aep_fuse_r[range(len(Lid_record)), range(len(Lid_record))].reshape(len(aep_fuse_r), 1)
	ranks_r = (probs >= 0).sum(axis=1)
	truth_r =  np.where(ranks_r==1)
	truths_r = truth_r[0].tolist()
	ind_r = np.argmax(probs, axis= 1)

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap_r = maxes-maxes1
	gap_r = gap_r

	MR, H10, MRR = cal_performance(ranks_r, top=10)
	_, H1, _ = cal_performance(ranks_r, top=1)

	print('to be evaluated... ' + str(len(ranks_r)))

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	return ind, gap, truths, ranks,  ind_r, gap_r, truths_r, ranks_r

def get_combine_hits_select_correct1(vec, name_vec, test_pair):
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
	ranks = (probs >= 0).sum(axis=1)

	truth =  np.where(ranks==1)
	truths = truth[0].tolist()
	ind = np.argmax(probs, axis= 1)
	#ind = np.append(ind, np.array(cannotmactch))

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap = maxes-maxes1


	MR, H10, MRR = cal_performance(ranks, top=10)
	_, H1, _ = cal_performance(ranks, top=1)

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	# right
	probs = aep_fuse_r - aep_fuse_r[range(len(Lid_record)), range(len(Lid_record))].reshape(len(aep_fuse_r), 1)
	ranks_r = (probs >= 0).sum(axis=1)
	truth_r =  np.where(ranks_r==1)
	truths_r = truth_r[0].tolist()
	ind_r = np.argmax(probs, axis= 1)

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap_r = maxes-maxes1

	MR, H10, MRR = cal_performance(ranks_r, top=10)
	_, H1, _ = cal_performance(ranks_r, top=1)

	print('to be evaluated... ' + str(len(ranks_r)))

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	return ind, gap, truths, ranks,  ind_r, gap_r, truths_r, ranks_r

# new measure
def solely_measure(vec, test_pair, dim):
	Lvec = tf.placeholder(tf.float32, [None, dim])
	Rvec = tf.placeholder(tf.float32, [None, dim])

	he = tf.nn.l2_normalize(Lvec, dim=-1) #??? 规范化啊
	norm_e_em = tf.nn.l2_normalize(Rvec, dim=-1)
	aep = tf.matmul(he, tf.transpose(norm_e_em))

	sess = tf.Session()
	Lv = np.array([vec[e1] for e1, e2 in test_pair])
	Lid_record = np.array([e1 for e1, e2 in test_pair])
	Rv = np.array([vec[e2] for e1, e2 in test_pair])
	Rid_record = np.array([e2 for e1, e2 in test_pair])
	aep_fuse = sess.run(aep, feed_dict = {Lvec: Lv, Rvec: Rv})

	probs = aep_fuse - aep_fuse[range(len(Lid_record)), range(len(Lid_record))].reshape(len(aep_fuse), 1)
	# only rank those who have correspondings... cause the set is distorted for those above max_correct
	ranks = (probs >= 0).sum(axis=1)
	print('to be evaluated... ' + str(len(ranks)))

	MR, H10, MRR = cal_performance(ranks, top=10)
	_, H1, _ = cal_performance(ranks, top=1)

	msg = 'Soly measure: Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

# old measure
def get_hits_select_ct(vec, test_pair, outfile, id2fre, theta_3, top_k=(1, 10, 50, 100)):
	Lvec = np.array([vec[e1] for e1, e2 in test_pair])
	Lid_record = np.array([e1 for e1, e2 in test_pair])
	Rvec = np.array([vec[e2] for e1, e2 in test_pair])
	Rid_record = np.array([e2 for e1, e2 in test_pair])
	sim = cdist(Lvec, Rvec, metric='cityblock')
	top_lr = [0] * len(top_k)
	mrr_sum_l = 0
	left2right = dict()
	for i in range(Lvec.shape[0]):
		rank = sim[i, :].argsort()
		lid = Lid_record[i]
		rid = Rid_record[rank[0]]
		left2right[lid] = rid
		rank_index = np.where(rank == i)[0][0]
		mrr_sum_l = mrr_sum_l + 1.0/(rank_index+1)
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_lr[j] += 1
	mrr_sum_r = 0
	top_rl = [0] * len(top_k)
	right2left = dict()
	for i in range(Rvec.shape[0]):
		rank = sim[:, i].argsort()
		rid = Rid_record[i]
		lid = Lid_record[rank[0]]
		right2left[rid] = lid
		rank_index = np.where(rank == i)[0][0]
		mrr_sum_r = mrr_sum_r + 1.0/(rank_index+1)
		for j in range(len(top_k)):
			if rank_index < top_k[j]:
				top_rl[j] += 1
	print('For each left:')
	outfile.write('For each left:\n')
	for i in range(len(top_lr)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
		outfile.write('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100) + '\n')
	outfile.write("MRR: " + str(mrr_sum_l/len(test_pair)) + '\n')
	print('For each right:')
	outfile.write('For each right:\n')
	for i in range(len(top_rl)):
		print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
		outfile.write('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100) + '\n')
	outfile.write("MRR: " + str(mrr_sum_r/len(test_pair)) + '\n')
	outfile.flush()
	matched = []
	for item in left2right:
		if left2right[item] not in right2left:
			print('wtf')
		else:
			if right2left[left2right[item]] == item:
				if id2fre[str(item)] >= theta_3:
					matched.append((item, left2right[item]))
	#print(matched)
	print(len(matched))
	outfile.write(str(len(matched)) + '\n\n')
	outfile.flush()
	print()

	return matched

def get_combine_hits_classification(vec, name_vec, test_pair, dicrank, max_correct):
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
	Lid_record = np.array([e1 for e1, e2 in test_pair]) # record the ids of inputs
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

	aep_fuse_rank = copy.deepcopy(aep_fuse)
	aep_fuse_rank = -aep_fuse_rank
	top_ids = aep_fuse_rank.argsort(axis = 1)[:,:10]# top_ids seem to be the positions, instead of the actual ids...

	aep_fuse_rank_r = copy.deepcopy(aep_fuse_r)
	aep_fuse_rank_r = -aep_fuse_rank_r
	top_ids_r = aep_fuse_rank_r.argsort(axis = 1)[:,:10]# top_ids seem to be the positions, instead of the actual ids...

	# 用top_ids 到　aep和aep_n中取值
	# 也可以在这里进行for and obtain those values... and get the labels
	feature = []
	label = []

	for item in range(len(Lid_record)): # 1- 1050
		ids = top_ids[item] # 1-20
		aep_item = aep[item][ids] # 1-20
		aep_n_item = aep_n[item][ids]
		for ii in range(len(ids)):
			feature.append([aep_item[ii], aep_n_item[ii]])
			if ids[ii] == item:
				label.append(1)
			else:
				label.append(0)

	# for item in range(len(Rid_record)): # 1- 1050
	# 	ids = top_ids_r[item] # 1-20
	# 	aep_item = aep.T[item][ids] # 1-20
	# 	aep_n_item = aep_n.T[item][ids]
	# 	for ii in range(len(ids)):
	# 		feature.append([aep_item[ii], aep_n_item[ii]])
	# 		if ids[ii] == item:
	# 			label.append(1)
	# 		else:
	# 			label.append(0)

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap = maxes-maxes1
	gap = gap[:max_correct]

	### pre
	pre = []
	for ent in dicrank.keys():
		pre.append(dicrank[ent])
	pre = np.array(pre)
	ranks_new = np.append(ranks, pre)
	print('to be evaluated... ' + str(len(ranks_new)))

	MR, H10, MRR = cal_performance(ranks_new, top=10)
	_, H1, _ = cal_performance(ranks_new, top=1)

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	# right
	probs = aep_fuse_r - aep_fuse_r[range(len(Lid_record)), range(len(Lid_record))].reshape(len(aep_fuse_r), 1)
	ranks_r = (probs >= 0).sum(axis=1)[:max_correct]
	truth_r =  np.where(ranks_r==1)
	truths_r = truth_r[0].tolist()
	ind_r = np.argmax(probs, axis= 1)[:max_correct]

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap_r = maxes-maxes1
	gap_r = gap_r[:max_correct]

	ranks_r = np.append(ranks_r, cannotmactch)

	MR, H10, MRR = cal_performance(ranks_r, top=10)
	_, H1, _ = cal_performance(ranks_r, top=1)

	print('to be evaluated... ' + str(len(ranks_r)))

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	return ind, gap, truths, ranks,  ind_r, gap_r, truths_r, ranks_r, feature, label

def get_combine_hits_classification_v1(vec, name_vec, test_pair, dicrank, max_correct):
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
	Lid_record = np.array([e1 for e1, e2 in test_pair]) # record the ids of inputs
	Rv = np.array([vec[e2] for e1, e2 in test_pair])
	Rid_record = np.array([e2 for e1, e2 in test_pair])

	Lv_ne = np.array([name_vec[e1] for e1, e2 in test_pair])
	Rv_ne = np.array([name_vec[e2] for e1, e2 in test_pair])
	aep = sess.run(aep, feed_dict = {Lvec: Lv, Rvec: Rv})
	aep_r = aep.T
	aep_n = sess.run(aep_n, feed_dict = {Lvec_ne: Lv_ne, Rvec_ne: Rv_ne})
	aep_n_r = aep_n.T
	aep_fuse = aep*0.3+ aep_n*0.7
	aep_fuse_r = aep_fuse.T


	# top in seperated matrix and get combined scores.... for further techniques..

	aep_rank = copy.deepcopy(aep)
	aep_rank = -aep_rank
	top_ids_stru = aep_rank.argsort(axis = 1)[:,:3]# top_ids seem to be the positions, instead of the actual ids...
	aep_rank.sort(axis = 1)
	top_scores_stru = aep_rank[:,:5]
	print(top_scores_stru)
	aep_n_rank = copy.deepcopy(aep_n)
	aep_n_rank = -aep_n_rank
	top_ids_name = aep_n_rank.argsort(axis = 1)[:,:3]# top_ids seem to be the positions, instead of the actual ids...
	aep_n_rank.sort(axis = 1)
	top_scores_name = aep_n_rank[:,:5]
	print(top_scores_name)

	aep_rank_r = copy.deepcopy(aep_r)
	aep_rank_r = -aep_rank_r
	top_ids_stru_r = aep_rank_r.argsort(axis = 1)[:,:3]# top_ids seem to be the positions, instead of the actual ids...
	aep_n_rank_r = copy.deepcopy(aep_n_r)
	aep_n_rank_r = -aep_n_rank_r
	top_ids_name_r = aep_n_rank_r.argsort(axis = 1)[:,:3]# top_ids seem to be the positions, instead of the actual ids...

	probs = aep_fuse - aep_fuse[range(len(Lid_record)), range(len(Lid_record))].reshape(len(aep_fuse), 1)
	# only rank those who have correspondings... cause the set is distorted for those above max_correct
	ranks = (probs >= 0).sum(axis=1)[:max_correct]
	cannotmactch = [10000]* (len(test_pair) - max_correct) #### 好烦啊，这个其实不知道他已经没有对象了，但是为了方便计算才从全局角度知道没有对象，每一顿还得加入训练。。。
	cannotmactch = np.array(cannotmactch)
	ranks = np.append(ranks, cannotmactch)

	truth =  np.where(ranks==1)
	truths = truth[0].tolist()
	ind = np.argmax(probs, axis= 1)[:max_correct]

	# 用top_ids 到　aep和aep_n中取值
	# 也可以在这里进行for and obtain those values... and get the labels
	feature = []
	label = []
	offset = []

	for item in range(len(Lid_record)): # 1- 1050
		ids_stru = top_ids_stru[item]
		ids_name = top_ids_name[item]
		ids = []
		for iddd in ids_stru:
			if aep[item,iddd] > 0.8:
				ids.append(iddd)
		for iddd in ids_name:
			if aep_n[item,iddd] > 0.6:
				if iddd not in ids:
					ids.append(iddd)
		offset.append(len(ids))
		if len(ids) > 0:
			aep_item = aep[item][ids] # 1-20
			aep_n_item = aep_n[item][ids]
			for ii in range(len(ids)):
				feature.append([aep_item[ii], aep_n_item[ii]])
				if ids[ii] == item:
					label.append(1)
				else:
					label.append(0)

	for item in range(len(Rid_record)): # 1- 1050
		ids_stru = top_ids_stru_r[item]
		ids_name = top_ids_name_r[item]
		ids = []
		for iddd in ids_stru:
			if aep.T[item,iddd] > 0.8:
				ids.append(iddd)
		for iddd in ids_name:
			if aep_n.T[item,iddd] > 0.6:
				if iddd not in ids:
					ids.append(iddd)
		offset.append(len(ids))
		if len(ids) > 0:
			aep_item = aep.T[item][ids] # 1-20
			aep_n_item = aep_n.T[item][ids]
			for ii in range(len(ids)):
				feature.append([aep_item[ii], aep_n_item[ii]])
				if ids[ii] == item:
					label.append(1)
				else:
					label.append(0)

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap = maxes-maxes1
	gap = gap[:max_correct]

	### pre
	pre = []
	for ent in dicrank.keys():
		pre.append(dicrank[ent])
	pre = np.array(pre)
	ranks_new = np.append(ranks, pre)
	print('to be evaluated... ' + str(len(ranks_new)))

	MR, H10, MRR = cal_performance(ranks_new, top=10)
	_, H1, _ = cal_performance(ranks_new, top=1)

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	# right
	probs = aep_fuse_r - aep_fuse_r[range(len(Lid_record)), range(len(Lid_record))].reshape(len(aep_fuse_r), 1)
	ranks_r = (probs >= 0).sum(axis=1)[:max_correct]
	truth_r =  np.where(ranks_r==1)
	truths_r = truth_r[0].tolist()
	ind_r = np.argmax(probs, axis= 1)[:max_correct]

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap_r = maxes-maxes1
	gap_r = gap_r[:max_correct]

	ranks_r = np.append(ranks_r, cannotmactch)

	MR, H10, MRR = cal_performance(ranks_r, top=10)
	_, H1, _ = cal_performance(ranks_r, top=1)

	print('to be evaluated... ' + str(len(ranks_r)))

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	return ind, gap, truths, ranks,  ind_r, gap_r, truths_r, ranks_r, feature, label, offset


def ins(aep, aep_n, aep_r, aep_n_r ):
	aep_rank = copy.deepcopy(aep)
	aep_rank = -aep_rank
	top_ids_stru = aep_rank.argsort(axis = 1)[:,:3]# top_ids seem to be the positions, instead of the actual ids...
	results_stru = aep_rank.argsort(axis = 1)[:,0]
	#print(results_stru)
	aep_rank.sort(axis = 1)
	top_scores_stru = -aep_rank[:,0]
	#print(top_scores_stru)
	# in the mean time, get the best matches


	aep_n_rank = copy.deepcopy(aep_n)
	aep_n_rank = -aep_n_rank
	top_ids_name = aep_n_rank.argsort(axis = 1)[:,:3]# top_ids seem to be the positions, instead of the actual ids...
	results_text = aep_n_rank.argsort(axis = 1)[:,0]
	#print(results_text)
	aep_n_rank.sort(axis = 1)
	top_scores_name = -aep_n_rank[:,0]
	#print(top_scores_name)


	# overlapped macthes
	overlapped = np.where(results_stru ==results_text)[0] # record the indexes where their values are equal...
	#print(overlapped)
	#print(len(overlapped))

	minstru = min(top_scores_stru[overlapped])
	mintext = min(top_scores_name[overlapped])
	#print(minstru)
	#print(mintext)

	# text first...
	text_conf_index = np.where(top_scores_stru >= minstru)[0]
	#print(len(text_conf_index))
	text_conf_index = overlapped
	results = results_stru[text_conf_index]
	result_dic = {text_conf_index[i]: results[i] for i in range(len(text_conf_index))}
	#print(result_dic)
	correct = 0
	for i in result_dic:
		if result_dic[i] == i:
			correct += 1
	#print(correct)

	aep_new = np.delete(aep, text_conf_index, axis = 0)
	aep_new = np.delete(aep_new, results, axis = 1)
	aep_n_new = np.delete(aep_n, text_conf_index, axis = 0)
	aep_n_new = np.delete(aep_n_new, results, axis = 1)

	aep_rank_r = copy.deepcopy(aep_r)
	aep_rank_r = -aep_rank_r
	top_ids_stru_r = aep_rank_r.argsort(axis = 1)[:,:3]# top_ids seem to be the positions, instead of the actual ids...
	aep_n_rank_r = copy.deepcopy(aep_n_r)
	aep_n_rank_r = -aep_n_rank_r
	top_ids_name_r = aep_n_rank_r.argsort(axis = 1)[:,:3]# top_ids seem to be the positions, instead of the actual ids...

	return top_ids_stru, top_ids_name, top_ids_stru_r, top_ids_name_r, aep_new, aep_n_new

def get_combine_hits_classification_v2(vec, name_vec, test_pair, dicrank, max_correct):
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
	Lid_record = np.array([e1 for e1, e2 in test_pair]) # record the ids of inputs
	Rv = np.array([vec[e2] for e1, e2 in test_pair])
	Rid_record = np.array([e2 for e1, e2 in test_pair])

	Lv_ne = np.array([name_vec[e1] for e1, e2 in test_pair])
	Rv_ne = np.array([name_vec[e2] for e1, e2 in test_pair])
	aep = sess.run(aep, feed_dict = {Lvec: Lv, Rvec: Rv})
	aep_r = aep.T
	aep_n = sess.run(aep_n, feed_dict = {Lvec_ne: Lv_ne, Rvec_ne: Rv_ne})
	aep_n_r = aep_n.T
	aep_fuse = aep*0.3+ aep_n*0.7
	aep_fuse_r = aep_fuse.T

	# top in seperated matrix and get combined scores.... for further techniques..
	top_ids_stru, top_ids_name, top_ids_stru_r, top_ids_name_r, aep_new, aep_n_new = ins(aep, aep_n, aep_r, aep_n_r)


	probs = aep_fuse - aep_fuse[range(len(Lid_record)), range(len(Lid_record))].reshape(len(aep_fuse), 1)
	# only rank those who have correspondings... cause the set is distorted for those above max_correct
	ranks = (probs >= 0).sum(axis=1)[:max_correct]
	cannotmactch = [10000]* (len(test_pair) - max_correct) #### 好烦啊，这个其实不知道他已经没有对象了，但是为了方便计算才从全局角度知道没有对象，每一顿还得加入训练。。。
	cannotmactch = np.array(cannotmactch)
	ranks = np.append(ranks, cannotmactch)

	truth =  np.where(ranks==1)
	truths = truth[0].tolist()
	ind = np.argmax(probs, axis= 1)[:max_correct]

	# 用top_ids 到　aep和aep_n中取值
	# 也可以在这里进行for and obtain those values... and get the labels
	feature = []
	label = []
	offset = []

	for item in range(len(Lid_record)): # 1- 1050
		ids_stru = top_ids_stru[item]
		ids_name = top_ids_name[item]
		ids = []
		for iddd in ids_stru:
			if aep[item,iddd] > 0.8:
				ids.append(iddd)
		for iddd in ids_name:
			if aep_n[item,iddd] > 0.6:
				if iddd not in ids:
					ids.append(iddd)
		offset.append(len(ids))
		if len(ids) > 0:
			aep_item = aep[item][ids] # 1-20
			aep_n_item = aep_n[item][ids]
			for ii in range(len(ids)):
				feature.append([aep_item[ii], aep_n_item[ii]])
				if ids[ii] == item:
					label.append(1)
				else:
					label.append(0)

	for item in range(len(Rid_record)): # 1- 1050
		ids_stru = top_ids_stru_r[item]
		ids_name = top_ids_name_r[item]
		ids = []
		for iddd in ids_stru:
			if aep.T[item,iddd] > 0.8:
				ids.append(iddd)
		for iddd in ids_name:
			if aep_n.T[item,iddd] > 0.6:
				if iddd not in ids:
					ids.append(iddd)
		offset.append(len(ids))
		if len(ids) > 0:
			aep_item = aep.T[item][ids] # 1-20
			aep_n_item = aep_n.T[item][ids]
			for ii in range(len(ids)):
				feature.append([aep_item[ii], aep_n_item[ii]])
				if ids[ii] == item:
					label.append(1)
				else:
					label.append(0)

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap = maxes-maxes1
	gap = gap[:max_correct]

	### pre
	pre = []
	for ent in dicrank.keys():
		pre.append(dicrank[ent])
	pre = np.array(pre)
	ranks_new = np.append(ranks, pre)
	print('to be evaluated... ' + str(len(ranks_new)))

	MR, H10, MRR = cal_performance(ranks_new, top=10)
	_, H1, _ = cal_performance(ranks_new, top=1)

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	# right
	probs = aep_fuse_r - aep_fuse_r[range(len(Lid_record)), range(len(Lid_record))].reshape(len(aep_fuse_r), 1)
	ranks_r = (probs >= 0).sum(axis=1)[:max_correct]
	truth_r =  np.where(ranks_r==1)
	truths_r = truth_r[0].tolist()
	ind_r = np.argmax(probs, axis= 1)[:max_correct]

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap_r = maxes-maxes1
	gap_r = gap_r[:max_correct]

	ranks_r = np.append(ranks_r, cannotmactch)

	MR, H10, MRR = cal_performance(ranks_r, top=10)
	_, H1, _ = cal_performance(ranks_r, top=1)

	print('to be evaluated... ' + str(len(ranks_r)))

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)


	top_ids_stru, top_ids_name, top_ids_stru_r, top_ids_name_r, aep_new, aep_n_new = ins(aep_new, aep_n_new, aep_r, aep_n_r)
	top_ids_stru, top_ids_name, top_ids_stru_r, top_ids_name_r, aep_new, aep_n_new = ins(aep_new, aep_n_new, aep_r, aep_n_r)
	top_ids_stru, top_ids_name, top_ids_stru_r, top_ids_name_r, aep_new, aep_n_new = ins(aep_new, aep_n_new, aep_r, aep_n_r)

	return ind, gap, truths, ranks,  ind_r, gap_r, truths_r, ranks_r, feature, label, offset

def get_combine_hits_classification_v3(vec, name_vec, test_pair, dicrank, max_correct):
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
	Lid_record = np.array([e1 for e1, e2 in test_pair]) # record the ids of inputs
	Rv = np.array([vec[e2] for e1, e2 in test_pair])
	Rid_record = np.array([e2 for e1, e2 in test_pair])

	Lv_ne = np.array([name_vec[e1] for e1, e2 in test_pair])
	Rv_ne = np.array([name_vec[e2] for e1, e2 in test_pair])
	aep = sess.run(aep, feed_dict = {Lvec: Lv, Rvec: Rv})
	aep_r = aep.T
	aep_n = sess.run(aep_n, feed_dict = {Lvec_ne: Lv_ne, Rvec_ne: Rv_ne})
	aep_n_r = aep_n.T


	# left2right and right2left for determining weight of features
	aep_rank = copy.deepcopy(aep)
	aep_rank = -aep_rank
	top_ids_stru = aep_rank.argsort(axis = 1)[:,:3]# top_ids seem to be the positions, instead of the actual ids...
	results_stru = aep_rank.argsort(axis = 1)[:,0]
	print(results_stru)
	aep_rank.sort(axis = 1)
	top_scores_stru = -aep_rank[:,0]
	print(top_scores_stru)
	left2right = {i: results_stru[i] for i in range(len(results_stru))}

	aep_rank_r = copy.deepcopy(aep_r)
	aep_rank_r = -aep_rank_r
	top_ids_stru_r = aep_rank_r.argsort(axis = 1)[:,:3]# top_ids seem to be the positions, instead of the actual ids...
	results_stru_r = aep_rank_r.argsort(axis = 1)[:,0]
	print(results_stru_r)
	aep_rank_r.sort(axis = 1)
	top_scores_stru_r = -aep_rank_r[:,0]
	print(top_scores_stru_r)
	right2left = {i: results_stru_r[i] for i in range(len(results_stru_r))}

	confident = dict()
	for item in left2right:
		if right2left[left2right[item]] == item:
			confident[item] = left2right[item]
	#print(confident)
	print('Confi in stru: ' + str(len(confident)))
	correct = 0
	for i in confident:
		if confident[i] == i:
			correct += 1
	print('Correct in stru: ' + str(correct))


	###
	aep_n_rank = copy.deepcopy(aep_n)
	aep_n_rank = -aep_n_rank
	top_ids_name = aep_n_rank.argsort(axis = 1)[:,:3]# top_ids seem to be the positions, instead of the actual ids...
	results_name = aep_n_rank.argsort(axis = 1)[:,0]
	print(results_name)
	aep_n_rank.sort(axis = 1)
	top_scores_name = -aep_n_rank[:,0]
	print(top_scores_name)
	left2right_n = {i: results_name[i] for i in range(len(results_name))}

	aep_n_rank_r = copy.deepcopy(aep_n_r)
	aep_n_rank_r = -aep_n_rank_r
	top_ids_name_r = aep_n_rank_r.argsort(axis = 1)[:,:3]# top_ids seem to be the positions, instead of the actual ids...
	results_name_r = aep_n_rank_r.argsort(axis = 1)[:,0]
	print(results_name_r)
	right2left_n = {i: results_name_r[i] for i in range(len(results_name_r))}

	confident_n = dict()
	for item in left2right_n:
		if right2left_n[left2right_n[item]] == item:
			confident_n[item] = left2right_n[item]
	#print(confident_n)
	print('Confi in text: ' + str(len(confident_n)))

	correct = 0
	for i in confident_n:
		if confident_n[i] == i:
			correct += 1
	print('Correct in text: ' + str(correct))
	# weight asigning


	matchedtrue = 0
	wei1 = 0
	sum_s = 0
	scores = []
	for item in confident:
		sum_s = sum_s + top_scores_stru[item]
		if item in confident_n and confident[item] == confident_n[item]:
			if item == confident[item]:
				matchedtrue += 1
			scores.append(top_scores_stru[item])
			wei1 += 0
		else:
			#if top_scores_stru[item]>=0.85:
				wei1 += 1
	wei2 = 0
	sum_n = 0
	for item in confident_n:
		if item in confident and confident_n[item] == confident[item]:
			sum_n = sum_n + top_scores_name[item]
			wei2 += 0
		else:
			if top_scores_name[item]>=0:
				wei2 += 1
	#print(float(sum_n)/float(len(confident_n)-wei2))
	print('Overlapped ' +  str(len(confident)-wei1) + ' True: ' + str(matchedtrue))
	print('Stru Sep Num: ' + str(wei1) + ' stru w: ' + str(float(wei1)/(wei1+wei2)))
	print('Text Sep Num: ' + str(wei2) + ' text w: ' + str(float(wei2)/(wei1+wei2)))
	print()
	wei1_f = float(wei1)/(wei1+wei2)
	wei2_f = float(wei2)/(wei1+wei2)


	aep_fuse = aep*wei1_f + aep_n*wei2_f
	#aep_fuse = aep*0.3 + aep_n*0.7
	aep_fuse_r = aep_fuse.T
	# top in seperated matrix and get combined scores.... for further techniques..
	top_ids_stru, top_ids_name, top_ids_stru_r, top_ids_name_r, aep_new, aep_n_new = ins(aep, aep_n, aep_r, aep_n_r)


	probs = aep_fuse - aep_fuse[range(len(Lid_record)), range(len(Lid_record))].reshape(len(aep_fuse), 1)
	# only rank those who have correspondings... cause the set is distorted for those above max_correct
	ranks = (probs >= 0).sum(axis=1)[:max_correct]
	cannotmactch = [10000]* (len(test_pair) - max_correct) #### 好烦啊，这个其实不知道他已经没有对象了，但是为了方便计算才从全局角度知道没有对象，每一顿还得加入训练。。。
	cannotmactch = np.array(cannotmactch)
	ranks = np.append(ranks, cannotmactch)

	truth =  np.where(ranks==1)
	truths = truth[0].tolist()
	ind = np.argmax(probs, axis= 1)[:max_correct]

	# 用top_ids 到　aep和aep_n中取值
	# 也可以在这里进行for and obtain those values... and get the labels
	feature = []
	label = []
	offset = []

	for item in range(len(Lid_record)): # 1- 1050
		ids_stru = top_ids_stru[item]
		ids_name = top_ids_name[item]
		ids = []
		for iddd in ids_stru:
			if aep[item,iddd] > 0.8:
				ids.append(iddd)
		for iddd in ids_name:
			if aep_n[item,iddd] > 0.6:
				if iddd not in ids:
					ids.append(iddd)
		offset.append(len(ids))
		if len(ids) > 0:
			aep_item = aep[item][ids] # 1-20
			aep_n_item = aep_n[item][ids]
			for ii in range(len(ids)):
				feature.append([aep_item[ii], aep_n_item[ii]])
				if ids[ii] == item:
					label.append(1)
				else:
					label.append(0)

	for item in range(len(Rid_record)): # 1- 1050
		ids_stru = top_ids_stru_r[item]
		ids_name = top_ids_name_r[item]
		ids = []
		for iddd in ids_stru:
			if aep.T[item,iddd] > 0.8:
				ids.append(iddd)
		for iddd in ids_name:
			if aep_n.T[item,iddd] > 0.6:
				if iddd not in ids:
					ids.append(iddd)
		offset.append(len(ids))
		if len(ids) > 0:
			aep_item = aep.T[item][ids] # 1-20
			aep_n_item = aep_n.T[item][ids]
			for ii in range(len(ids)):
				feature.append([aep_item[ii], aep_n_item[ii]])
				if ids[ii] == item:
					label.append(1)
				else:
					label.append(0)

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap = maxes-maxes1
	gap = gap[:max_correct]

	### pre
	pre = []
	for ent in dicrank.keys():
		pre.append(dicrank[ent])
	pre = np.array(pre)
	ranks_new = np.append(ranks, pre)
	print('to be evaluated... ' + str(len(ranks_new)))

	MR, H10, MRR = cal_performance(ranks_new, top=10)
	_, H1, _ = cal_performance(ranks_new, top=1)

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	# right
	probs = aep_fuse_r - aep_fuse_r[range(len(Lid_record)), range(len(Lid_record))].reshape(len(aep_fuse_r), 1)
	ranks_r = (probs >= 0).sum(axis=1)[:max_correct]
	truth_r =  np.where(ranks_r==1)
	truths_r = truth_r[0].tolist()
	ind_r = np.argmax(probs, axis= 1)[:max_correct]

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap_r = maxes-maxes1
	gap_r = gap_r[:max_correct]

	ranks_r = np.append(ranks_r, cannotmactch)

	MR, H10, MRR = cal_performance(ranks_r, top=10)
	_, H1, _ = cal_performance(ranks_r, top=1)

	print('to be evaluated... ' + str(len(ranks_r)))

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	return ind, gap, truths, ranks,  ind_r, gap_r, truths_r, ranks_r, feature, label, offset

# a seperated dic for mapping to original id

def ite(aep_fuse, aep_fuse_r, dic_row, dic_col):
	aep_fuse_rank = copy.deepcopy(aep_fuse)
	aep_fuse_rank = -aep_fuse_rank
	results_stru = aep_fuse_rank.argsort(axis = 1)[:,0]
	print(results_stru)
	aep_fuse_rank.sort(axis = 1)
	top_scores_stru = -aep_fuse_rank[:,0]
	print(top_scores_stru)
	left2right = {i: results_stru[i] for i in range(len(results_stru))}

	aep_fuse_rank_r = copy.deepcopy(aep_fuse_r)
	aep_fuse_rank_r = -aep_fuse_rank_r
	results_stru_r = aep_fuse_rank_r.argsort(axis = 1)[:,0]
	print(results_stru_r)
	aep_fuse_rank_r.sort(axis = 1)
	top_scores_stru_r = -aep_fuse_rank_r[:,0]
	print(top_scores_stru_r)
	right2left = {i: results_stru_r[i] for i in range(len(results_stru_r))}

	confident = dict()
	row = []
	col = []
	for item in left2right:
		if right2left[left2right[item]] == item:
			if top_scores_stru[item] > 0:
				confident[item] = left2right[item]
				row.append(item)
				col.append(left2right[item])
	#print(confident)
	print('Confi in fuse: ' + str(len(confident)))
	correct = 0
	for i in confident:
		if dic_col[confident[i]] == dic_row[i]:
			correct += 1
	print('Correct in fuse: ' + str(correct))

	# after removal, need to define a mapping function to map column/rows indexes to the origional indexes
	newind_row = 0
	new2old_row = dict()
	newind_col = 0
	new2old_col = dict()
	for item in range(aep_fuse.shape[0]):
		if item not in row:
			new2old_row[newind_row] = dic_row[item] # dic_row item not just item # item is one-hop map while dic_row...
			newind_row += 1
		if item not in col:
			new2old_col[newind_col] = dic_col[item]
			newind_col += 1
	print(len(new2old_row))
	print(len(new2old_col))

	aep_fuse_new = np.delete(aep_fuse, row, axis=0)
	aep_fuse_new = np.delete(aep_fuse_new, col, axis=1)
	aep_fuse_r_new = aep_fuse_new.T

	return aep_fuse_new, aep_fuse_r_new, len(confident), correct, new2old_row, new2old_col

def get_combine_hits_select_correct_v3(vec, name_vec, test_pair, dicrank, max_correct):
	Lvec = tf.placeholder(tf.float32, [None, 900])
	Rvec = tf.placeholder(tf.float32, [None, 900])
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
	aep_fuse = aep*0.26822+ aep_n*0.73178
	aep_fuse_r = aep_fuse.T

	total_matched = 0
	total_true = 0
	dic_row = {i:i for i in range(9450)}
	dic_col = {i:i for i in range(9450)}
	aep_fuse_new = copy.deepcopy(aep_fuse)
	aep_fuse_r_new = copy.deepcopy(aep_fuse_r)
	while total_matched < 9450:
		print(dic_row)
		print(dic_col)
		aep_fuse_new, aep_fuse_r_new, matched, matched_true, dic_row, dic_col = ite(aep_fuse_new, aep_fuse_r_new, dic_row, dic_col)
		total_matched += matched
		total_true += matched_true
		print('Total Match ' + str(total_matched))
		print('Total Match True ' + str(total_true))

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

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap = maxes-maxes1
	gap = gap[:max_correct]

	### pre
	pre = []
	for ent in dicrank.keys():
		pre.append(dicrank[ent])
	pre = np.array(pre)
	ranks_new = np.append(ranks, pre)
	print('to be evaluated... ' + str(len(ranks_new)))

	MR, H10, MRR = cal_performance(ranks_new, top=10)
	_, H1, _ = cal_performance(ranks_new, top=1)

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	# right
	probs = aep_fuse_r - aep_fuse_r[range(len(Lid_record)), range(len(Lid_record))].reshape(len(aep_fuse_r), 1)
	ranks_r = (probs >= 0).sum(axis=1)[:max_correct]
	truth_r =  np.where(ranks_r==1)
	truths_r = truth_r[0].tolist()
	ind_r = np.argmax(probs, axis= 1)[:max_correct]

	maxes = np.max(probs, axis= 1)
	probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
	maxes1 = np.max(probs, axis= 1)
	gap_r = maxes-maxes1
	gap_r = gap_r[:max_correct]

	ranks_r = np.append(ranks_r, cannotmactch)

	MR, H10, MRR = cal_performance(ranks_r, top=10)
	_, H1, _ = cal_performance(ranks_r, top=1)

	print('to be evaluated... ' + str(len(ranks_r)))

	msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
	print('\n'+msg)

	return ind, gap, truths, ranks,  ind_r, gap_r, truths_r, ranks_r