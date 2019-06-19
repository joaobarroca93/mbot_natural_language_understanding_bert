#!/usr/bin/env python

from __future__ import print_function
 
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import keras
from keras.utils import to_categorical

import bert
from bert import run_classifier, optimization, tokenization


def merge_two_dicts(x, y):
	z = x.copy()
	z.update(y)
	return z

class NaturalLanguageUnderstanding(object):

	def __init__(self, bert_path, classifier_path, label_list, debug=False, max_num_preds=10, max_seq_length=30):

		if debug == False:
			tf.logging.set_verbosity(tf.logging.ERROR)

		self.max_seq_length = max_seq_length
		self.batch_size = max_num_preds

		self.bert_path = bert_path
		self.classifier_path = classifier_path

		print("Downloading tokenizer")
		self.tokenizer = self.create_tokenizer_from_hub_module()

		run_config = tf.estimator.RunConfig(model_dir=self.classifier_path)

		model_fn = self.model_fn_builder(label_list=label_list)

		params = {
			"batch_size": self.batch_size,
		}

		print("Creating model")
		self.estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params=params)

	def create_tokenizer_from_hub_module(self):

		with tf.Graph().as_default():
			bert_module = hub.Module(self.bert_path)
			tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
			with tf.Session() as sess:
				vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
													tokenization_info["do_lower_case"]])
		return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


	def create_model(self, is_predicting, input_ids, input_mask, segment_ids, labels, label_list, dropout_rate=2e-5,
				 learning_rate=0.1, num_train_steps=1, num_warmup_steps=1, use_tpu=False):
	
		bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
		
		with tf.variable_scope("bert", reuse=tf.AUTO_REUSE):
			# Create the BERT Module, with pooled_output as the output
			bert_module = hub.Module(self.bert_path, trainable=False)
			bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)

			# The output of the BERT module will be a vector "representing" the whole sentence, with a shape of (batch_size, hidden_size).
			bert_output = bert_outputs["pooled_output"]
			hidden_size = bert_output.shape[-1].value
			
		if not is_predicting:
			# Dropout helps prevent overfitting, if we are not predicting.
			bert_output = tf.nn.dropout(bert_output, rate=dropout_rate)
		
		""" Now I need to create multiple heads, each one containing a softmax for classification along a specific slot """
		
		num_labels_dict = { key: len(label_list[key]) for key in label_list.keys() }
		num_labels_list = [ len(label_list[key]) for key in label_list.keys() ]
		
		# Adds the multiple softmax layers.
		with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):

			output_weights = {
				slot: tf.get_variable( 'weights_%s' % slot, [num_labels, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02) ) 
								for slot, num_labels in zip(num_labels_dict.keys(), num_labels_dict.values()) }
			
			output_bias = {
					slot: tf.get_variable( "bias_%s" % slot, [num_labels], initializer=tf.zeros_initializer() )
							 for slot, num_labels in zip(num_labels_dict.keys(), num_labels_dict.values()) }
			
			logits_dict = {
				slot: tf.nn.bias_add( tf.matmul(bert_output, output_weights[slot], transpose_b=True), output_bias[slot], name='bias_add_%s' % slot)
							 for slot in num_labels_dict.keys() }
			
			log_probs_dict = { slot: tf.nn.log_softmax(logits_dict[slot], axis=-1, name='log_softmax_%s' % slot) for slot in num_labels_dict.keys() }
			
			probs_dict = { slot: tf.nn.softmax(logits_dict[slot], axis=-1, name='softmax_%s' % slot) for slot in num_labels_dict.keys() }
			
		
		if not is_predicting:
		
			with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):

				one_hot_labels_dict = { key: tf.one_hot(labels[key], depth=num_labels_dict[key], dtype=tf.float32, name='one_hot_%s' % key) for key in label_list.keys() } 

				per_example_loss_dict = {
					slot: -tf.reduce_sum(one_hot_labels_dict[slot] * log_probs_dict[slot], axis=-1, name='per_example_loss_%s' % slot)
										for slot in num_labels_dict.keys() }

				per_slot_loss_dict = { slot: tf.reduce_mean(per_example_loss_dict[slot], name='per_slot_loss_%s' % slot) for slot in num_labels_dict.keys() }
				loss = tf.reduce_sum( tf.stack([ per_slot_loss_dict[slot] for slot in num_labels_dict.keys()], axis=0), name='total_loss')
			
		with tf.variable_scope("prediction", reuse=tf.AUTO_REUSE):

			predictions = { slot: tf.squeeze( tf.argmax(probs_dict[slot], axis=-1, output_type=tf.int32) ) for slot in num_labels_dict.keys() }

			if is_predicting:
				return (predictions, probs_dict)
			
		with tf.variable_scope("training", reuse=tf.AUTO_REUSE):
			
			train_ops = tf.group([
				bert.optimization.create_optimizer(per_slot_loss_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=use_tpu)
									for per_slot_loss_loss in per_slot_loss_dict.values()], name='training_ops')
				
		return (train_ops, loss, predictions, probs_dict)


	# model_fn_builder actually creates our model function
	# using the passed parameters for num_labels, learning_rate, etc.
	def model_fn_builder(self, label_list, dropout_rate=0.1, learning_rate=2e-5, num_train_steps=1, num_warmup_steps=1, use_tpu=False):
		

		def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
			
			is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

			input_ids = features["input_ids"]
			input_mask = features["input_mask"]
			segment_ids = features["segment_ids"]
			# dummy assignment, when the model is predicting.
			label_ids = features["label_ids"]
			
			# label list is a dict, whose keys are the slots and whose values are lists
			# of all the possible values for each slot.
			num_labels = len(label_list['destination']) 
			
			if not is_predicting:
				# the label_ids is a dict, whose keys are the slots and whose values are the label_id for that specific slot.
				# the label_id is an integer representing the value for a specific slot.
				label_ids = {key: features[key] for key in label_list.keys()}


			# TRAIN and EVAL
			if not is_predicting:

				train_ops, loss, predictions_dict, probs_dict = self.create_model(is_predicting, input_ids, input_mask, segment_ids,
																		label_ids, label_list)

				
				"""CHANGE NUM_LABELS TO LABEL_LIST !!!"""
				# Calculate evaluation metrics. 
				def metric_fn(label_ids, predictions_dict, probs_dict, num_labels):

					def l2_norm(label_ids, probs, num_labels):
						one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)
						l2 = tf.norm( (one_hot_labels - probs), ord='euclidean' , axis=0)
						m_l2, update_l2_op = tf.metrics.mean(l2)
						return m_l2, update_l2_op   

					accuracy = { slot+"_acc": tf.metrics.accuracy(label_ids[slot], predictions_dict[slot]) for slot in label_list.keys() }
					accuracy["acc"] = tf.metrics.mean(tf.reduce_prod( list( accuracy.values() ), axis=0 ))
					
					
					l2 = { slot+"_l2": l2_norm( label_ids[slot], probs_dict[slot], len(label_list[slot]) ) for slot in label_list.keys() }
					l2["l2"] = tf.metrics.mean( list( l2.values() ) )

					return merge_two_dicts(accuracy, l2)

				eval_metrics = metric_fn(label_ids, predictions_dict, probs_dict, num_labels)
				

				if mode == tf.estimator.ModeKeys.TRAIN:
					return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_ops)
				else:
					return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)
			else:
				predictions_dict, probs_dict = self.create_model(is_predicting, input_ids, input_mask, segment_ids, label_ids, label_list)

				#predictions = merge_two_dicts(probs_dict, predictions_dict)
				
				return tf.estimator.EstimatorSpec(mode, predictions=probs_dict)

		# Return the actual model function in the closure
		return model_fn

	def predict(self, pred_sentences, label_list):
		input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = "none") for x in pred_sentences] # here, "" is just a dummy label
		input_features = run_classifier.convert_examples_to_features(input_examples, ["none"], self.max_seq_length, self.tokenizer)
		predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=self.max_seq_length, is_training=False, drop_remainder=False)
		predictions = self.estimator.predict(predict_input_fn)
		return [(sentence, prediction) for sentence, prediction in zip(pred_sentences, predictions)]


if __name__ == '__main__':

	# load from ontology
	label_list = LABEL_LIST


	BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
	OUTPUT_DIR = 'classifier_model'

	nlu_object = NaturalLanguageUnderstanding(
		bert_path=BERT_MODEL_HUB,
		classifier_path=OUTPUT_DIR,
		label_list=label_list
	)

	pred_sentences = [
		"take the banana from the cuisine",
		"look for the melon near the kitchen",
		"search the kitchen for an apple"
	]

	preds = nlu_object.predict(pred_sentences, label_list)



	"""
	for pred in preds:
		print("{:>40} -> d-type={}, intent={}, object={}, person={}, source={}, destination={}".format(
			pred[0], label_list['d-type'][pred[1]['d-type']], label_list['intent'][pred[1]['intent']], label_list['object'][pred[1]['object']], label_list['person'][pred[1]['person']],
			label_list['source'][pred[1]['source']], label_list['destination'][pred[1]['destination']] ))
	"""


	for pred in preds:

		predictions = {}
		total_prob = 1
		for label in label_keys:
			index = np.argmax(pred[1][label])
			pred_label = label_list[label][index]
			prob = pred[1][label][index]
			predictions[label] = [pred_label, prob]
			total_prob = total_prob * prob
		predictions["probability"] = total_prob


		print("{:>60} -> {}(".format(pred[0], predictions['d-type'][0]), end='')

		for slot in slots:
			value = predictions[slot][0]
			if not value is "none":
				print('{}={}'.format(slot, value), end=',')

		print(")[{:0.1}]".format(predictions['probability']))
