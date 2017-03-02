import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as img
from os import listdir
import time
start_time = time.time()

# Hyperparmeter

batch_size=16
learning_rate=1e-4
epoch=1000

def image_reader(path):
	imagesList = listdir(path)

	print("imagesList: %d" % len(imagesList))

	images = []
	for i in imagesList:
	    immg=img.open(path+i)
	    immg=immg.resize([320,180])
	    images.append(np.asarray(immg).astype(np.float32)/255.0)

	return imagesList, images

train_list, images_train = image_reader("./images/")
test_list, images_test = image_reader("./images_test/")

def residual_resize(tensor,num_outputs):
	layer1=tf.contrib.layers.batch_norm(tensor)
	layer1=tf.nn.relu(layer1)
	layer1=tf.contrib.layers.conv2d(layer1,num_outputs=num_outputs,kernel_size=3,stride=2,padding="SAME",activation_fn=None)
	layer2=tf.contrib.layers.batch_norm(layer1)
	layer2=tf.nn.relu(layer2)
	layer2=tf.contrib.layers.conv2d(layer2,num_outputs=num_outputs,kernel_size=3,stride=1,padding="SAME",activation_fn=None)

	return layer2

def residual_cell(tensor,num_outputs):
	input_layer=tensor
	layer1=tf.contrib.layers.batch_norm(input_layer)
	layer1=tf.nn.relu(layer1)
	layer1=tf.contrib.layers.conv2d(layer1,num_outputs=num_outputs,kernel_size=3,stride=1,padding="SAME",activation_fn=None)
	layer2=tf.contrib.layers.batch_norm(layer1)
	layer2=tf.nn.relu(layer2)
	layer2=tf.contrib.layers.conv2d(layer2,num_outputs=num_outputs,kernel_size=3,stride=1,padding="SAME",activation_fn=None)
	output=tf.concat([input_layer,layer2],axis=3)
	output=tf.contrib.layers.conv2d(output,num_outputs=num_outputs,kernel_size=1,stride=1,padding="SAME",activation_fn=None)

	return output

def residual_block(tensor,num_outputs,num):
	for i in range(num):
		tensor=residual_cell(tensor,num_outputs)

	return tensor

def decoder_block(tensor,num_outputs,last_layer=False):
	height = tf.shape(tensor)[1]
	width  = tf.shape(tensor)[2]

	if last_layer:
		layer0 = tf.contrib.layers.conv2d(tensor,num_outputs=4*num_outputs,kernel_size=3,stride=1,padding="SAME",activation_fn=None)
		layer0 = tf.nn.tanh(layer0)
		layer0 = tf.reshape(layer0,shape=[-1, 2*height, 2*width, num_outputs])
		return layer0

	else:
		layer0 = tf.contrib.layers.conv2d(tensor,num_outputs=4*num_outputs,kernel_size=3,stride=1,padding="SAME",activation_fn=None)
		layer0 = tf.contrib.layers.batch_norm(layer0)
		layer0 = tf.nn.relu(layer0)
		layer0 = tf.reshape(layer0,shape=[-1, 2*height, 2*width, num_outputs])
		return layer0

# Encoder

X = tf.placeholder(dtype=tf.float32,shape=[None,180,320,3])

layer0 = tf.contrib.layers.conv2d(X,num_outputs=64,kernel_size=7,stride=2)

layer1 = residual_block(layer0,64,3)  # None,90,160,64

layer2 = residual_resize(layer1,128)
layer2 = residual_block(layer2,128,3) # None,45,80,128

layer3 = residual_resize(layer2,256)
layer3 = residual_block(layer3,256,5) # None,23,40,256

layer4 = residual_resize(layer3,512)
layer4 = residual_block(layer4,512,2) # None,12,20,512

layer5 = tf.contrib.layers.conv2d(layer4, num_outputs=1024,kernel_size=2,stride=2,padding="SAME") 			# None,6,10,1024
layer5 = tf.contrib.layers.conv2d(layer5, num_outputs=1024,kernel_size=2,stride=2,padding="SAME") 			# None,3,5,1024
layer5 = tf.contrib.layers.conv2d(layer5, num_outputs=512,kernel_size=1,stride=1,padding="SAME")   			# None,3,5,512
layer5 = tf.contrib.layers.conv2d(layer5, num_outputs=256,kernel_size=1,stride=1,padding="SAME")  			# None,3,5,256
layer5 = tf.contrib.layers.conv2d(layer5, num_outputs=128,kernel_size=1,stride=1,padding="SAME")   			# None,3,5,128

encoded = tf.reshape(layer5, shape=[batch_size,3*5*128])

embedding_init = tf.Variable(tf.zeros([batch_size,3*5*128]),dtype=tf.float32, name="embedding")				# None, 3*5*128
embedding = embedding_init.assign(encoded)
embed_final=tf.Variable(tf.zeros([len(images_test),3*5*128]),dtype=tf.float32, name="embedding_total")

# Decoder

dec_layer0 = tf.reshape(encoded, shape=[batch_size,3,5,128])
dec_layer0 = tf.contrib.layers.conv2d(dec_layer0, num_outputs=256,kernel_size=1,stride=1,padding="SAME")
dec_layer0 = tf.contrib.layers.conv2d(dec_layer0, num_outputs=512,kernel_size=1,stride=1,padding="SAME")
dec_layer0 = tf.contrib.layers.conv2d(dec_layer0, num_outputs=1024,kernel_size=1,stride=1,padding="SAME")   # None,3,5,1024

dec_layer1 = decoder_block(dec_layer0,1024) 								  							    # None,6,10,1024

dec_layer2 = decoder_block(dec_layer1,512)  								  								# None,12,20,512

dec_layer3 = decoder_block(dec_layer2,256)  								  								# None,24,40,256
dec_layer3 = tf.slice(dec_layer3,begin=[0,0,0,0],size=[batch_size,23,40,256]) 								# None,23,40,256
 
dec_layer4 = decoder_block(dec_layer3,128) 									  								# None,46,80,128
dec_layer4 = tf.slice(dec_layer4,begin=[0,0,0,0],size=[batch_size,45,80,128])							    # None,45,80,128
		
dec_layer5 = decoder_block(dec_layer4,128) 									  								# None,90,160,64

dec_layer6 = decoder_block(dec_layer5,3,last_layer=True)									 								# None,180,320,3

decoded = dec_layer6

# Optimization

loss = tf.reduce_sum(tf.abs(decoded-X))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init=tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)
	embedding_acc=sess.run(embedding,feed_dict={X:images_test[0:batch_size]}) # embedding variable for test images

	try:
		saver.restore(sess=sess, save_path="./model/model.ckpt")
		print("\n--------model restored--------\n")
	except:
		print("\n--------model Not restored--------\n")
		pass


	# training
	

	print("\n--------Start Training--------\n")

	for i in range(epoch):
		acc_loss=0
		for j in range(int(len(images_train)/batch_size)):
			#print(j)
			image_feed=images_train[j*batch_size:j*batch_size+batch_size]
			_, loss_train, decode_train = sess.run([optimizer,loss,decoded],feed_dict={X:image_feed})
			acc_loss+=loss_train

		print("epoch: {} ,train loss: {}".format(i, acc_loss/len(images_train)))

		if i % 10 is 0 and i is not 0:
			saver.save(sess, './model/model.ckpt')
			print("\n--------model saved--------\n")
			plt.imsave("epoch:"+str(i)+"_"+train_list[0][:-3]+"png",decode_train[0])

	print("--- %s seconds ---" % (time.time() - start_time))
	
	
	# test


	print("\n--------Start Testing--------\n")

 
	for i in range(int(len(images_test)/batch_size)):
		image_feed=images_test[i*batch_size:i*batch_size+batch_size]
		encoder_test,embedding_test,loss_test,decode_test = sess.run([encoded,embedding,loss,decoded],feed_dict={X:image_feed})
		embedding_acc=tf.concat(axis=0,values=[embedding_acc,embedding_test])
		print(loss_test)
		plt.imsave(str(i)+"_"+test_list[i][:-3]+"png",decode_test[0])

	# total embedding variable save

	print("\n--------Start Embedding--------\n")

	sess.run(embed_final.assign(embedding_acc[batch_size:]))

	saver.save(sess, './model/model.ckpt')

	print("\n--------model saved--------")
