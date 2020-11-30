import tensorflow as tf


class Config:
	language = 'en_fr_15k_V1' # dbp_wd_15k_V1 | en_fr_15k_V1 | wk3l_60k/en_de
	e1 = 'data/' + language + '/mapping/0_3/ent_ids_1'
	e2 = 'data/' + language + '/mapping/0_3/ent_ids_2'
	kg1 = 'data/' + language + '/mapping/0_3/triples_1'
	kg2 = 'data/' + language + '/mapping/0_3/triples_2'
	ill = 'data/' + language + '/mapping/0_3/ref_ent_ids_all'
	# if language in ['zh_en', 'ja_en', 'fr_en']:
	# 	a1 = 'data/' + language + '/training_attrs_1'
	# 	a2 = 'data/' + language + '/training_attrs_2'
	# 	r1 = 'data/' + language + '/rel_ids_1'
	# 	r2 = 'data/' + language + '/rel_ids_2'
	# 	seed = 3  # 30% of seeds
	# elif language in ['dbp_yg', 'dbp_wd']:
	# 	r1 = 'data/' + language + '/rel_ids_1'
	# 	r2 = 'data/' + language + '/rel_ids_2'
	# 	seed = 3  # 30% of seeds
	# else:
	# 	if language == 'wk3l_60k/en_fr':
	# 		seed = 0.241  # 24.1% of seeds
	# 	else:
	# 		seed = 0.225  # 22.5% of seeds
	seed = 0.3

	epochs_se = 300
	se_dim = 300
	act_func = tf.nn.relu
	gamma = 3.0  # margin based loss
	k = 25  # number of negative samples for each positive one
	beta = 0.3  # weight of SE
	theta_1 = 100
	theta_2 = 100
	theta_3 = 10
	theta_4 = 10
	epsilon = 3
