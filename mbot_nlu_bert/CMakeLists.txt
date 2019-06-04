cmake_minimum_required(VERSION 2.8.3)
project(mbot_nlu_bert)

find_package(catkin REQUIRED COMPONENTS
	message_generation
	std_msgs
	rospy
)

add_message_files(
	DIRECTORY ros/msg
   	FILES
   		TurnInfo.msg
		InformSlot.msg
	  	DialogAct.msg
	  	DialogActArray.msg
	  	ASRHypothesis.msg
	  	ASRNBestList.msg
)

catkin_python_setup()

generate_messages(
	DEPENDENCIES
	std_msgs
)

catkin_package(CATKIN_DEPENDS
	message_runtime
)