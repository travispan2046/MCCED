import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D, Reshape, Conv3D, Flatten, RepeatVector
from keras import regularizers
import numpy as np
from keras.models import Model
import scipy.io as scio
import random
from keras.models import Model
from keras.layers import Input, LSTM, Dense, ConvLSTM2D, BatchNormalization, Conv3D, TimeDistributed,Activation
from sklearn import metrics
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.layers import Lambda
#import matplotlib.pyplot as plt
import argparse
import json
import datetime
import os
import TIED_ED
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))



def NN_predict(model,dataX,dataY):
    dataX = dataX.reshape(dataX.shape[0], dataX.shape[1], dataX.shape[2], dataX.shape[3], 1)
    dataY = dataY.reshape(dataY.shape[0], dataY.shape[1], dataY.shape[2], dataY.shape[3], 1)

    dataX_zeros=np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2], dataX.shape[3], 1))  # (smaple,1,20,51,1)
    dataX_decoder_zeros = np.zeros((dataX.shape[0], 1, dataX.shape[2], dataX.shape[3], 1))  # (smaple,1,20,51,1)

    if dataY.shape[1]==1:
        [predict_label1, predict_label2] = model.predict([dataX, dataX_zeros,dataX_decoder_zeros], batch_size=32, verbose=1)  #
    else:
        [predict_label1, predict_label2] = model.predict([dataX, dataX_zeros,dataX_decoder_zeros], batch_size=32, verbose=1)  #
    predict_label1 = predict_label1[:, ::-1, :, :, :]

    data_predict_NN = predict_label2.reshape(
        predict_label2.shape[0] * predict_label2.shape[1] * predict_label2.shape[2], predict_label2.shape[3])
    data_rcstr_NN = predict_label1.reshape(predict_label1.shape[0] * predict_label1.shape[1] * predict_label1.shape[2],
                                           predict_label1.shape[3])
    data_predict_true = dataY.reshape(dataY.shape[0] * dataY.shape[1] * dataY.shape[2], dataY.shape[3])
    data_rcstr_true = dataX.reshape(dataX.shape[0] * dataX.shape[1] * dataX.shape[2], dataX.shape[3])

    return data_predict_NN, data_rcstr_NN, data_predict_true, data_rcstr_true
def options_parser():
    parser = argparse.ArgumentParser(description='Train a neural network to handle real-valued data.')
    # meta-option
    parser.add_argument('--settings_file', help='json file of settings, overrides everything else', type=str, default='')
    return parser

def load_settings_from_file(settings):
    # settings.txt
    settings_path = "./"+settings['settings_file'] + ".txt"
    print('Loading settings from', settings_path)
    settings_loaded = json.load(open(settings_path, 'r',encoding='utf-8'))
    # check for settings missing in file
    return settings_loaded
# def weighted_loss()
def get_settings_and_files():
    parser = options_parser()
    settings = vars(parser.parse_args())

    if settings['settings_file']:
        settings = load_settings_from_file(settings)

    result_path = "../seconddata/" + settings['preprocessing_timepath'] + "ConvLstm预处理结果(conditional版本)/"
    list = os.listdir(result_path)
    total_result = []
    for i in range(0, len(list)):
        path = os.path.join(result_path, list[i])
        if os.path.isfile(path):
            total_result.append(np.load(path)['result'][()])
    choice_result = total_result[0]
    train_input = choice_result['train_input']
    train_predict = choice_result["train_predict"]
    test_input = choice_result["test_input"]
    test_predict = choice_result["test_predict"]
    ground_truth=choice_result["ground_truth"]
    params = settings
    return params,choice_result["tag"],ground_truth,train_input,train_predict,test_input,test_predict

def network(params,train_input,train_predict):

    ##############################################    ENCODER    #############################################
    # try:
    encoder_inputs = Input(shape=(train_input.shape[1], train_input.shape[2],train_input.shape[3], 1))
    # encoder_outputs = ConvLSTM2D(filters=params['filter'], kernel_size=(3, 3), padding='same', return_sequences=True)(encoder_inputs)
    # encoder_outputs = BatchNormalization()(encoder_outputs)
    # encoder_outputs = ConvLSTM2D(filters=params['filter'], kernel_size=(3, 3), padding='same', return_sequences=True)(encoder_outputs)

    encoder = ConvLSTM2D(filters=params['filter'], kernel_size=(3, 3), padding='same',return_sequences=True,return_state=True, kernel_initializer='he_uniform',activity_regularizer=regularizers.l1(params["regularizer"]))
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    state_h = BatchNormalization()(state_h)
    state_c=BatchNormalization()(state_c)
    # print(encoder_outputs,encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    # encoder_states = [state_h, state_c]  #copy的最近一个时刻的状态值

    ##############################################    Dictionary Meomory   ##############################################
    # memory layer
    memory_size = params["memory_size"]
    addressing = TIED_ED.DenseLayerAutoencoder([memory_size], use_bias=False, name="addressing_x")

    # encoder:c*memory_size
    # attention weight (?,memory_size)
    # decoder:memory_size*c
    # after getting attention, the latent_vector:
    flat_encoder_outputs = TimeDistributed(Flatten())(encoder_outputs)
    print("flat_encoder_outputs.shape", flat_encoder_outputs.shape)
    memoried_encoder_outputs0 = []
    for i in range(train_input.shape[1]):
        print("flat_encoder_outputs[:, i, :].shape", flat_encoder_outputs[:, i, :].shape)
        print("train_input.shape[2]*train_input.shape[3]*params",
              train_input.shape[2] * train_input.shape[3] * params["filter"])
        flat_encoder_outputs_i0 = Lambda(lambda x: x[:, i, :])(flat_encoder_outputs)
        addressed_state0 = addressing(flat_encoder_outputs_i0)
        addressed_state = Reshape((1, train_input.shape[2], train_input.shape[3], params["filter"]))(addressed_state0)
        memoried_encoder_outputs0.append(addressed_state)
    memoried_encoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(memoried_encoder_outputs0)

    # memoried_encoder_outputs=TimeDistributed(addressing)(flat_encoder_outputs)
    state_h_use=Lambda(lambda x: x[:, -1, :, :, :])(memoried_encoder_outputs)
    state_c1=Flatten()(state_c)
    state_c_use0=addressing(state_c1)
    state_c_use=Reshape((train_input.shape[2], train_input.shape[3], params["filter"]))(state_c_use0)

    ##############################################    PAST DECODER    ##############################################
    # Set up the decoder, using `encoder_states` as initial state.
    past_decoder_inputs = Input(shape=(train_input.shape[1], train_input.shape[2],train_input.shape[3], 1))
    decoder_lstm = ConvLSTM2D(filters=params['filter'], kernel_size=(3, 3), padding='same', return_sequences=True, kernel_initializer='he_uniform', name='decode_con')
    past_decoder_outputs = decoder_lstm([past_decoder_inputs, state_h_use, state_c_use])
    decoder_dense = Conv3D(filters=1, kernel_size=(3, 3, 3),activation=params['activation'],padding='same', kernel_initializer='he_uniform',data_format='channels_last')
    past_decoder_outputs = decoder_dense(past_decoder_outputs)


    ##############################################    FUTURE DECODER    ##############################################
    
    encoder_outputs_for_mul = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1, 4)), name="encoder_outputs_for_mul")

    context_vector_mul = Lambda(
        lambda x: K.permute_dimensions(K.batch_dot(x, atten_wgt_in_step_T, axes=[3, 4]), (0, 4, 1, 2, 3)),
        name="context_vector_mul")

    decoder_dense_pre = Conv2D(filters=1, kernel_size=(1, 1), activation=params['activation'], padding='same',
                               data_format='channels_last', kernel_initializer='he_uniform')

    ##########    attention      ##########
    # attention 1: get attention_weight
    def atten_wgt(EN_hidden_states, last_hidden_state):
        score_in_step = []
        EN_time_steps = train_input.shape[1]
        for j in range(EN_time_steps):
            score_in_step_j=K.sum(EN_hidden_states[:, j, :, :, :]*last_hidden_state,axis=[1,2,3])
            score_in_step.append(score_in_step_j)
        score_in_step=K.stack(score_in_step,axis=0)
        score_in_step=K.permute_dimensions(score_in_step, (1, 0))
        score_in_step=Reshape((EN_time_steps,1))(score_in_step)/params["scale_factor"]
        atten_wgt_in_step = K.softmax(score_in_step, axis=1)
        return atten_wgt_in_step
    get_atten_wgt_layer = Lambda(lambda x: atten_wgt(encoder_outputs, x), name="get_atten_wgt_layer")

    # attention 2: change attention_weight dimension
    def adjust_atten_wgt_in_step(x):
        x = K.permute_dimensions(x, (0, 2, 1))  # atten_wgt_in_step (?, 1, 3)
        x = K.expand_dims(x, axis=1)
        x = K.expand_dims(x, axis=1)
        x = K.repeat_elements(x, train_predict.shape[2], 1)
        x = K.repeat_elements(x, train_predict.shape[3], 2)
        return x
    adjust_atten_wgt_in_step_layer = Lambda(lambda x: adjust_atten_wgt_in_step(x), name="adjust_atten_wgt_in_step_layer")

    
    decoder_lstm_pre = ConvLSTM2D(filters=params['filter'], kernel_size=(3, 3), padding='same', return_state=True,
                                  return_sequences=False, kernel_initializer='he_uniform', name='decode_pre')
    BN3 = BatchNormalization(name="BN3")
    output_reshape = Reshape((1, train_input.shape[2], train_input.shape[3], 1), name="output_reshape")
    fu_decoder_inputs = Input(shape=(1, train_input.shape[2], train_input.shape[3], 1))
    future_decoder_input_concat = Lambda(lambda x: K.concatenate([x, context_vector], axis=-1),
                                         name="future_decoder_input_concat")
    decoder_addressing = TIED_ED.DenseLayerAutoencoder([memory_size], use_bias=False, name="addressing_y")
    
    ##########    train_priduct.shape[1] =1    ##########
    if train_predict.shape[1] == 1:
       
        '''<START> get context_vector'''
        memoried_encoder_outputs_T = encoder_outputs_for_mul(memoried_encoder_outputs)
        atten_wgt_in_step = get_atten_wgt_layer(state_h)
        # atten_wgt_in_step (?, 3, 1)
        atten_wgt_in_step_T = Lambda(lambda x: adjust_atten_wgt_in_step(x), name="adjust_atten_wgt_in_step")(
            atten_wgt_in_step)
        '''<END> '''
        
        context_vector = context_vector_mul(memoried_encoder_outputs_T)  # shape=(?, 40, 51, 64)
        future_decoder_outputs = context_vector
        future_decoder_outputs = decoder_dense_pre(future_decoder_outputs)
        future_decoder_outputs = Reshape((train_predict.shape[1], train_predict.shape[2], train_predict.shape[3], 1))(
            future_decoder_outputs)
        ##############################################    Model Compile    ##############################################
        COMPOSITE_ED = Model(inputs=[encoder_inputs, past_decoder_inputs],
                             outputs=[past_decoder_outputs, future_decoder_outputs])
        # COMPOSITE_ED.summary()
        rmsprop = optimizers.RMSprop(lr=params['lr'], rho=0.9, epsilon=None, decay=0.0)
        COMPOSITE_ED.compile(loss=[params['loss1'], params['loss2']], loss_weights=[0.0, 1.0], optimizer=rmsprop)

    ##########      train_priduct.shape[1] > 1    ##########
    else:
        # future_decoder_inputs = Input(shape=(
        # train_predict.shape[1], train_predict.shape[2], train_predict.shape[3], 1))  # get future_decoder_inputs

        all_outputs = []
        max_decoder_seq_length = train_predict.shape[1]
        state_in_h=state_h_use
        state_in_c=state_c_use

        
        fu_decoder_inputs_loop=fu_decoder_inputs
        for i in range(max_decoder_seq_length):
            # attention

            '''<START> get context_vector'''
            memoried_encoder_outputs_T = encoder_outputs_for_mul(memoried_encoder_outputs)  # 已经改成经过memory的encoder
            atten_wgt_in_step = get_atten_wgt_layer(state_in_h)
            # shape(?,3,1)
            atten_wgt_in_step_T = adjust_atten_wgt_in_step_layer(atten_wgt_in_step)
            # atten_wgt_in_step_T (?, 1, 1, 1, 3)
            # encoder_outputs_T(?, ?, ?, 3, ?)
            # batch_dot
            context_vector = context_vector_mul(memoried_encoder_outputs_T)  # shape=(?, 40, 51, 64)
            # shape=(?, 1, 40, 51, 64)
            '''<END> '''
            '''external memory in decoder'''
            
            concat_inputs = future_decoder_input_concat(fu_decoder_inputs_loop)
            concat_inputs = Flatten()(concat_inputs)

            addressed_concat_inputs=decoder_addressing(concat_inputs)
            addressed_concat_inputs=Reshape((1, train_predict.shape[2], train_predict.shape[3], params["filter"]+1))(addressed_concat_inputs)

            output, state_in_h, state_in_c = decoder_lstm_pre([addressed_concat_inputs, state_in_h, state_in_c])
            output = BN3(output)
            output = decoder_dense_pre(output)  # channel = 1
            output = output_reshape(output)
            print("output shape", output.shape)
            # output => fu_decoder_inputs
            fu_decoder_inputs_loop=output
            all_outputs.append(output)
            # Reinject the outputs as inputs for the next loop iteration
            # as well as update the states
        # Concatenate all predictions
        

        outputs_concat = Lambda(lambda x: K.concatenate(x, axis=1), name="outputs_concat")
        future_decoder_outputs = outputs_concat(all_outputs)
        
        outputs_reshape = Reshape((train_predict.shape[1], train_predict.shape[2], train_predict.shape[3], 1),
                                  name="outputs_reshape")
        future_decoder_outputs = outputs_reshape(future_decoder_outputs)
        COMPOSITE_ED = Model(inputs=[encoder_inputs, past_decoder_inputs,fu_decoder_inputs],outputs=[past_decoder_outputs, future_decoder_outputs])
        # COMPOSITE_ED.summary()
        rmsprop = optimizers.RMSprop(lr=params['lr'], rho=0.9, epsilon=None, decay=0.0)
        COMPOSITE_ED.compile(loss=[params['loss1'], params['loss2']], loss_weights=[0.0, 1.0],
                             optimizer=rmsprop)
        ##############################################    Model saving    ##############################################

    return COMPOSITE_ED

def model_operation(COMPOSITE_ED):
    params, tag, ground_truth, train_input, train_predict, test_input, test_predict=get_settings_and_files()
    train_zeros = np.zeros((train_input.shape[0], train_input.shape[1], train_input.shape[2], train_input.shape[3], 1))
    print("Start training")
    print(train_input.shape, train_predict.shape, test_input.shape, test_predict.shape)
    #(4807, 5, 20, 52) (4807, 5, 20, 52) (4498, 5, 20, 52) (4498, 5, 20, 52)


    
    permutation = np.random.permutation(train_input.shape[0])
    train_input = train_input[permutation, :]
    train_predict = train_predict[permutation,:]

    train_input = train_input.reshape(train_input.shape[0],train_input.shape[1],train_input.shape[2],train_input.shape[3],1)

    train_predict = train_predict.reshape(train_predict.shape[0],train_predict.shape[1],train_predict.shape[2],train_predict.shape[3],1)
    train_zeros = np.zeros((train_input.shape[0], train_input.shape[1], train_input.shape[2], train_input.shape[3], 1))
    train_decoder_zeros = np.zeros((train_input.shape[0],1,train_input.shape[2],train_input.shape[3],1)) #(smaple,1,20,51,1)

    print('Trying', params)
    train_input_rev = train_input[:, ::-1, :, :, :]

    if train_predict.shape[1] == 1:
        filepath = pathm + nowTime + 'model-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')
        # COMPOSITE_ED.summary()
        history = COMPOSITE_ED.fit([train_input, train_zeros], [train_input_rev, train_predict],
                                   nb_epoch=params['nb_epochs'], batch_size=settings['batch_size'],
                                   callbacks=[checkpoint], validation_split=0.2)
    else:
        filepath = pathm + nowTime + 'model-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')
        #COMPOSITE_ED.summary()
        print("train_decoder_zeros",train_decoder_zeros.shape)
        history=COMPOSITE_ED.fit([train_input, train_zeros,train_decoder_zeros], [train_input_rev, train_predict], nb_epoch=params['nb_epochs'], batch_size=settings['batch_size'], callbacks=[checkpoint], validation_split=0.2)

    np.savez(pathm+nowTime+"history_data.npz",epoch=history.epoch,history=history.history)
    

    ##############################################    save data    ##############################################

    train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true =NN_predict(COMPOSITE_ED,train_input,train_predict)
    test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true = NN_predict(COMPOSITE_ED,test_input,test_predict)
    my_dict = {"tag": tag,
                "train_predict_NN": train_predict_NN,
               "train_rcstr_NN": train_rcstr_NN,
               "test_predict_NN": test_predict_NN,
               "test_rcstr_NN": test_rcstr_NN,
               "train_predict_true": train_predict_true,
               "train_rcstr_true": train_rcstr_true,
               "test_predict_true": test_predict_true,
               "test_rcstr_true": test_rcstr_true,
               "ground_truth": ground_truth}
    return my_dict





if __name__=="__main__":
    parser = options_parser()
    settings = vars(parser.parse_args())

    if settings['settings_file']:
        settings = load_settings_from_file(settings)

    nowTime = datetime.datetime.now().strftime('%m_%d_%H_%M')
    result_path = "../seconddata/"+settings['preprocessing_timepath']+"ConvLstm_preprocessing(conditional)/"
    list = os.listdir(result_path)
    total_result = []
    for i in range(0, len(list)):
        path = os.path.join(result_path, list[i])
        if os.path.isfile(path):
            total_result.append(np.load(path)['result'][()])

    pathc = "../resultdata/" + nowTime + "conditional_training" + "(conditional)/"
    os.makedirs(pathc)
    pathm = pathc+"models/"
    os.makedirs(pathm)

    for i in range(len(total_result)):
        pathm_i = pathm + total_result[i]['tag']+"models/"
        os.makedirs(pathm_i)
        start_NN_time = datetime.datetime.now()
        COMPOSITE_ED = network(settings, total_result[i]["train_input"],total_result[i]["train_predict"])
        NN_choice_result=model_operation(COMPOSITE_ED)
        end_NN_time = datetime.datetime.now()
        m, s = divmod(((end_NN_time - start_NN_time).total_seconds()), 60)
        h, m = divmod(m, 60)
        print(NN_choice_result["tag"]  + "time" + str(h) + "hours" + str(m) + "minutes" + str(s) + "seconds")
        time_result = pathc + nowTime + "conditional_training" + NN_choice_result['tag']
        np.savez(time_result, result=NN_choice_result)



