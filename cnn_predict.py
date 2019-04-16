from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import csv

def write_result(name, predictions):
    """
    """
    if predictions is None:
        raise Exception('need predictions')

    predictions = predictions.flatten()

    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    path = os.path.join('./results/', name)

    with open(path, 'wt', encoding='utf-8', newline='') as csv_target_file:
        target_writer = csv.writer(csv_target_file, lineterminator='\n')

        header = [
            'user_id',
            'time_slot_0', 'time_slot_1', 'time_slot_2', 'time_slot_3',
            'time_slot_4', 'time_slot_5', 'time_slot_6', 'time_slot_7',
            'time_slot_8', 'time_slot_9', 'time_slot_10', 'time_slot_11',
            'time_slot_12', 'time_slot_13', 'time_slot_14', 'time_slot_15',
            'time_slot_16', 'time_slot_17', 'time_slot_18', 'time_slot_19',
            'time_slot_20', 'time_slot_21', 'time_slot_22', 'time_slot_23',
            'time_slot_24', 'time_slot_25', 'time_slot_26', 'time_slot_27',
        ]

        target_writer.writerow(header)

        for i in range(0, len(predictions), 28):
            # NOTE: 57159 is the offset of user ids
            userid = [57159 + i // 28]
            labels = predictions[i:i+28].tolist()

            target_writer.writerow(userid + labels)

tf.reset_default_graph()
with tf.device('/CPU:0'):
    sess = tf.Session()

    # 載入模型
    saver = tf.train.import_meta_graph('./model_2/cnn_0.8575981259346008_0.2087835669517517.ckpt-8401.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model_2/'))

    # 載入graph
    y = tf.get_collection('outputs')[0]
    graph = tf.get_default_graph()

    xs = graph.get_tensor_by_name("inputs/xs:0")
    keep_prob = graph.get_tensor_by_name("inputs/keep_prob:0")

    dataset = np.load('./datasets/v0_eigens.npz')

    # NOTE: read features of test set
    test_eigens = dataset['issue_eigens'][:, :-28].reshape(-1, 896)
    # test_eigens = test_eigens[:, -28:]

    result = sess.run(y, feed_dict = {xs: test_eigens, keep_prob: 1})
    # result3 = sess.run(tf.where(tf.less(result, 0.03), tf.zeros_like(result), tf.ones_like(result)))
    # result2 = sess.run(tf.where(tf.less(result, 0.02), tf.zeros_like(result), tf.ones_like(result)))
    
    write_result('result_cnn_3.csv', result)
    # write_result('result_7_002.csv', result2)