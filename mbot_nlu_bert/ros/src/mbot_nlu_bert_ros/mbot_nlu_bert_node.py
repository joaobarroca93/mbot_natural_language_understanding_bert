#!/usr/bin/env python

from __future__ import print_function

import copy
import json
import os
import string

import numpy as np

import rospy
import rospkg

from std_msgs.msg import String
from mbot_nlu_bert.msg import InformSlot, DialogAct, DialogActArray, ASRHypothesis, ASRNBestList
from mbot_nlu_bert.mbot_nlu_bert_common_v3 import NaturalLanguageUnderstanding


# parameters 
# '~loop_rate'


#SUB_TOPIC_NAME = '/recognized_speech'
#PUB_TOPIC_NAME = '/dialogue_acts'
#BERT_MODEL_DIR = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
#OUTPUT_DIR = "/ros/src/mbot_nlu_bert_ros/multi_head_classifier_model_v2"
#ONTHOLOGY_PATH = "/ros/src/mbot_nlu_bert_ros/onthology.json"
#SLOTS = ['intent', 'object', 'destination', 'source', 'person']


"""
Description: This function helps logging parameters as debug verbosity messages.

Inputs:
	- param_dict: a dictionary whose keys are the parameter's names and values the parameter's values.
"""
def logdebug_param(param_dict):
	[ rospy.logdebug( '{:<25}\t{}'.format(param[0], param[1]) ) for param in param_dict.items() ]


class NLUNode(object):

	def __init__(self, debug=False):

		# get useful parameters, and if any of them doesn't exist, use the default value
		rate 			= rospy.get_param('~loop_rate', 10.0)
		#slots 			= rospy.get_param('~slots', ['destination'])
		slots 			= rospy.get_param('~slots', ['intent', 'person', 'object', 'source', 'destination'])
		node_name 		= rospy.get_param('~node_name', 'natural_language_understanding')
		d_acts_topic 	= rospy.get_param('~dialogue_acts_topic_name', '/dialogue_acts')
		ontology_name 	= rospy.get_param('~ontology_full_name', 'ros/src/mbot_nlu_bert_ros/ontology.json')
		output_dir 		= rospy.get_param('~classifier_model_full_path', 'ros/src/mbot_nlu_bert_ros/multi_head_classifier_model_v2')
		bert_model_dir 	= rospy.get_param('~bert_model_full_path', 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1')
		max_seq_length  = rospy.get_param('~max_seq_length', 20)
		max_num_preds	= rospy.get_param('~max_num_preds', 10)

		# initializes the node (if debug, initializes in debug mode)
		if debug == True:
			rospy.init_node(node_name, anonymous=False, log_level=rospy.DEBUG)
			rospy.loginfo("%s node created [DEBUG MODE]" % node_name)
		else:
			rospy.init_node(node_name, anonymous=False)
			rospy.loginfo("%s node created" % node_name)

		# set parameters to make sure all parameters are set to the parameter server
		rospy.set_param('~loop_rate', rate)
		rospy.set_param('~slots', slots)
		rospy.set_param('~node_name', node_name)
		rospy.set_param('~dialogue_acts_topic_name', d_acts_topic)
		rospy.set_param('~ontology_full_name', ontology_name)
		rospy.set_param('~classifier_model_full_path', output_dir)
		rospy.set_param('~bert_model_full_path', bert_model_dir)
		rospy.set_param('~max_seq_length', max_seq_length)
		rospy.set_param('~max_num_preds', max_num_preds)

		rospy.logdebug('=== NODE PRIVATE PARAMETERS ============')
		logdebug_param(rospy.get_param(node_name))

		rospack = rospkg.RosPack()
		# get useful paths
		generic_path = rospack.get_path("mbot_nlu_bert")
		classifier_path = os.path.join( generic_path, output_dir )
		ontology_path = os.path.join( generic_path, ontology_name )
		logdebug_param({'generic_path': generic_path, 'classifier_path': classifier_path, 'onthology_full_path': ontology_path})

		with open(ontology_path, "r") as f:
			self.label_list = json.load(f)
		rospy.logdebug('=== DIALOGUE LABEL LIST ============')
		rospy.logdebug(self.label_list)

		self.nlu_object = NaturalLanguageUnderstanding(bert_path=bert_model_dir, classifier_path=classifier_path,
			label_list=self.label_list, debug=debug, max_num_preds=max_num_preds, max_seq_length=max_seq_length)
		rospy.loginfo('natural language understanding object created')

		self.nlu_request_received = False
		self.asr_n_best_list = None
		self.slots = slots
		self.label_keys = slots + ['d-type']
		self.loop_rate = rospy.Rate(rate)

		n_best_topic = 'asr_n_best_list'
		rospy.Subscriber(n_best_topic, ASRNBestList, self.nluCallback, queue_size=1)
		rospy.loginfo("subscribed to topic %s", n_best_topic)

		self.pub_sentence_recog = rospy.Publisher(d_acts_topic, DialogActArray, queue_size=1)
		rospy.loginfo("publishing to topic %s", d_acts_topic)

		rospy.loginfo("%s initialization completed! Ready to accept requests" % node_name)

	def nluCallback(self, msg):

		rospy.loginfo('[Message received]')
		rospy.logdebug('{}'.format(msg))

		self.asr_n_best_list = msg
		self.nlu_request_received = True

	def preprocess_sentences(self, sentences):

		no_punct_sent = [
			sentence.translate(string.maketrans("",""), string.punctuation)
		for sentence in sentences ]

		return no_punct_sent



	def begin(self):

		while not rospy.is_shutdown():

			if self.nlu_request_received == True:

				rospy.loginfo('[Handling message]' )
				self.nlu_request_received = False

				pred_sentences = [hypothesis.transcript for hypothesis in self.asr_n_best_list.hypothesis]
				#pred_sentences = pred_sentences[0:3]
				
				# preprocess user utterance hypothesis before feeding the semantic decoder
				pred_sentences = self.preprocess_sentences(pred_sentences)
				# compute probability distribution of each hypothesis through softmax of confidence scores
				confs = [hypothesis.confidence for hypothesis in self.asr_n_best_list.hypothesis]
				probs = np.exp(confs) / np.sum(np.exp(confs))

				rospy.logdebug('sentences to predict: {}'.format(pred_sentences))

				rospy.loginfo('Beginning Predictions!')
				preds = self.nlu_object.predict(pred_sentences, self.label_list)

				"""
				for pred in preds:
					rospy.logdebug(pred)
				"""

				# ================================ Organize the conditonal probabilities ================================
				rospy.logdebug('\n======= CONDITIONAL PROBS =======\n')
				predictions = []
				for j, pred in enumerate(preds):
					slot_value_dict = {label: [(value, 0) for value in self.label_list[label]] for label in self.label_keys}
					for label in self.label_keys:
						for i, value in enumerate(self.label_list[label]):
							new_prob = pred[1][label][i]
							slot_value_dict[label][i]=( (value, new_prob) )
					
					[ slot_value_dict[label].sort(key=lambda x: x[1], reverse=True) for label in self.label_keys ]
					rospy.logdebug('======= {}[{}] ======='.format(pred_sentences[j], probs[j]))
					rospy.logdebug(slot_value_dict)
					predictions.append(slot_value_dict)

				# ================================ Compute the joint probabilities ================================
				predictions = []
				rospy.logdebug('\n======= JOINT PROBS =======\n')
				for j, pred in enumerate(preds):
					slot_value_dict = {label: [(value, 0) for value in self.label_list[label]] for label in self.label_keys}
					for label in self.label_keys:
						for i, value in enumerate(self.label_list[label]):
							new_prob = pred[1][label][i] * probs[j]
							slot_value_dict[label][i]=( (value, new_prob) )
					
					[ slot_value_dict[label].sort(key=lambda x: x[1], reverse=True) for label in self.label_keys ]
					rospy.logdebug('======= {} ======='.format(pred_sentences[j]))
					rospy.logdebug(slot_value_dict)
					predictions.append(slot_value_dict)

				
				# ================================ Computes marginal probabilities ================================
				# using the sum rule, i.e. summing over all the user utterance hypothesis, we compute the marginal
				# porbabilities.
				rospy.logdebug('\n======= MARGINAL PROBS =======\n')
				predictions = {label: [(value, 0) for value in self.label_list[label]] for label in self.label_keys}
				for j, pred in enumerate(preds):
					for label in self.label_keys:
						for i, value in enumerate(self.label_list[label]):
							new_prob = predictions[label][i][1] + pred[1][label][i] * probs[j]
							predictions[label][i]=( (value, new_prob) )
							
				[ predictions[label].sort(key=lambda x: x[1], reverse=True) for label in self.label_keys ]
				rospy.logdebug(predictions)
				
 

				# ================================ Creates DialogActs to publish ================================
				dialogue_act_array_msg = DialogActArray()

				for j, pred in enumerate(preds):
					predictions = {}
					total_prob = 1
					for label in self.label_keys:
						index = np.argmax(pred[1][label])
						pred_label = self.label_list[label][index]
						prob = pred[1][label][index] * probs[j]
						predictions[label] = [pred_label, prob]
						total_prob = total_prob * prob
					predictions['probability'] = total_prob

					dialogue_act_msg = DialogAct()
					dialogue_act_msg.dtype = predictions['d-type'][0]
					print("{} -> {}(".format(pred[0], predictions['d-type'][0]), end='')

					dialogue_act_msg.d_type_probability = predictions['d-type'][1]
					print('[{:0.3}]'.format(dialogue_act_msg.d_type_probability))

					for slot in self.slots:
						value = predictions[slot][0]
						prob = predictions[slot][1]
						dialogue_act_msg.slots.append(InformSlot(slot=slot, value=value, probability=prob))
						if not value is "none":
							print('{}={}[{}]'.format(slot, value, prob), end=',')
					print()

					dialogue_act_array_msg.dialogue_acts.append(copy.deepcopy(dialogue_act_msg))

				self.pub_sentence_recog.publish(dialogue_act_array_msg)
			

			self.loop_rate.sleep()

def main():

	nlu_node = NLUNode(debug=True)
	nlu_node.begin()