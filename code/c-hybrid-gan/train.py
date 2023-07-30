"""
Script to train the C-Hybrid-MLE & C-Hybrid-GAN model on test data.
C-Hybrid-MLE is the model obtained at the end of pre-training."""

import tensorflow as tf
import pandas as pd

from generator import *
from discriminator import *
from drivers import *
from utils import *

import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    
    """# Data"""
    
    print("Data: \n")

    # create label encoders for each song attribute

    data = np.load(FULL_DATA_PATH)

    print("Shape of full data:: ", data.shape)

    y = data[:, :SONG_LENGTH * NUM_SONG_FEATURES]              # 60 cols
    y = np.reshape(y, (-1, SONG_LENGTH, NUM_SONG_FEATURES))    # 60 = 20 x 3: 1 pitch, 2 duration, 3 rest

    print("Shape of melody data:", y.shape)

    y_p = y[:, :, 0]
    y_d = y[:, :, 1]
    y_r = y[:, :, 2]

    print("Shape of pitch data:", y_p.shape)
    print("Shape of duration data:", y_d.shape)
    print("Shape of rest data:", y_d.shape)

    print("Number of unique pitches, durations and rests present in data:")
    NUM_P_TOKENS = create_categorical_2d_encoder(y_p, P_LE_PATH)
    NUM_D_TOKENS = create_categorical_2d_encoder(y_d, D_LE_PATH)
    NUM_R_TOKENS = create_categorical_2d_encoder(y_r, R_LE_PATH)

    NUM_TOKENS= [NUM_P_TOKENS, NUM_D_TOKENS, NUM_R_TOKENS]
    LE_PATHS  = [P_LE_PATH, D_LE_PATH, R_LE_PATH]

    # load train, validation and test data

    x_train, y_train_dat_attr, y_train = load_data(TRAIN_DATA_PATH,
                                                   LE_PATHS,
                                                   SONG_LENGTH,
                                                   NUM_SONG_FEATURES,
                                                   NUM_META_FEATURES)

    x_valid, y_valid_dat_attr, y_valid = load_data(VALID_DATA_PATH,
                                                   LE_PATHS,
                                                   SONG_LENGTH,
                                                   NUM_SONG_FEATURES,
                                                   NUM_META_FEATURES,
                                                   convert_to_tensor=True)

    x_test,  y_test_dat_attr,  y_test  = load_data(TEST_DATA_PATH,
                                                   LE_PATHS,
                                                   SONG_LENGTH,
                                                   NUM_SONG_FEATURES,
                                                   NUM_META_FEATURES,
                                                   convert_to_tensor=True)

    TRAIN_LEN = len(x_train)
    VALID_LEN = len(x_valid)
    TEST_LEN  = len(x_test)

    STEPS_PER_EPOCH_TRAIN = np.ceil(TRAIN_LEN/BATCH_SIZE)
    print('Steps per epoch train: ', STEPS_PER_EPOCH_TRAIN)

    # create train dataset object

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(TRAIN_LEN)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)

    print("================================================================ \n " )
    """# Training"""

    # Set seed for reproducibility

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    ## Initialise Model

    # Initialise generator model
    g_map_model = Generator_mapping(400, 'relu', Attention_EMB_UNITS, Attention_PROJ_UNITS, Attention_EMB_DROPOUT_RATE,
                                    Attention_PROJ_DROPOUT_RATE, Attention_MEM_SLOTS, Attention_HEAD_SIZE, Attention_NUM_HEADS, Attention_NUM_BLOCKS,
                                    NUM_TOKENS, NUM_META_FEATURES)
    g_model = Generator(
        G_EMB_UNITS, G_PROJ_UNITS, G_EMB_DROPOUT_RATE, G_PROJ_DROPOUT_RATE,
        G_MEM_SLOTS, G_HEAD_SIZE, G_NUM_HEADS, G_NUM_BLOCKS, NUM_TOKENS,
        NUM_META_FEATURES)

    # Initialise discriminator
    d_map_model = Discriminator_mapping(20, 'relu')

    d_model = Discriminator(D_EMB_UNITS, D_PROJ_UNITS, D_EMB_DROPOUT_RATE, D_PROJ_DROPOUT_RATE,
                            D_MEM_SLOTS, D_HEAD_SIZE, D_NUM_HEADS, D_NUM_BLOCKS, NUM_TOKENS,
                            NUM_META_FEATURES)

    ## Initialise Optimizer

    # Initialise optimizer for pretraining
    pre_train_g_opt = tf.keras.optimizers.Adam(
        PRE_TRAIN_LR, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Initialise optimizer for adversarial training
    adv_train_g_opt = tf.keras.optimizers.Adam(ADV_TRAIN_G_LR, beta_1=0.9, beta_2=0.999)
    adv_train_d_opt = tf.keras.optimizers.Adam(ADV_TRAIN_D_LR, beta_1=0.9, beta_2=0.999)

    ## Initialise Driver

    # Initialise pre-train driver
    pre_train_driver = PreTrainDriver(g_model, 
                                      pre_train_g_opt,
                                      MAX_GRAD_NORM,
                                      NUM_TOKENS)

    # Initialise adversarial driver
    adv_train_driver = AdversarialDriver(g_model,
                                         g_map_model,
                                         d_model,
                                         d_map_model,
                                         adv_train_g_opt,
                                         adv_train_d_opt,
                                         TEMP_MAX,
                                         STEPS_PER_EPOCH_TRAIN,
                                         ADV_TRAIN_EPOCHS,
                                         MAX_GRAD_NORM, 
                                         NUM_TOKENS)

    ## Setup Checkpoint

    # Setup checkpoint for pretraining
    pre_train_ckpt = tf.train.Checkpoint(g_model=g_model,
                                         pre_train_g_opt=pre_train_g_opt)

    pre_train_ckpt_manager = tf.train.CheckpointManager(
        pre_train_ckpt, PRE_TRAIN_CKPT_PATH, max_to_keep=PRE_TRAIN_EPOCHS)

    # Setup checkpoint for adversarial training
    adv_train_ckpt = tf.train.Checkpoint(g_model=g_model,
                                         g_map_model=g_map_model,
                                         d_model=d_model,
                                         d_map_model=d_map_model,
                                         adv_train_g_opt=adv_train_g_opt,
                                         adv_train_d_opt=adv_train_d_opt)



    adv_train_ckpt_manager = tf.train.CheckpointManager(
        adv_train_ckpt, ADV_TRAIN_CKPT_PATH, max_to_keep=ADV_TRAIN_EPOCHS)

    ## Setup Logging

    train_summary_writer = tf.summary.create_file_writer(TRAIN_LOG_DIR)
    valid_summary_writer = tf.summary.create_file_writer(VALID_LOG_DIR)
    test_summary_writer  = tf.summary.create_file_writer(TEST_LOG_DIR)
    
    if REUSE_TEST_LOGS:
        # load test logs from previous run
        try: 
            test_logs = pd.read_csv(TEST_LOGS_FILENAME).to_dict('records')
        except FileNotFoundError: 
            test_logs = []
    else:
        test_logs = []

    # """## Tensorboard setup"""

    # ## Setup Tensorboard

    # %tensorboard --logdir LOG_DIR

    # # Share tensorboard for remote vizualizaton

    # !tensorboard dev upload --logdir LOG_DIR

    """## Training Loop"""

    # Training

    ## PreTraining

    print('\n ***** Starting PreTraining ***** \n')

    for epoch in range(PRE_TRAIN_EPOCHS):
        start = time.time()

        # log test metrics for current epoch
        logs = {'seed' : SEED,
                'epoch': epoch,
                'base' : 'rmc'}

        total_train_p_loss = 0
        total_train_d_loss = 0
        total_train_r_loss = 0

        for batch, inp in enumerate(train_dataset.take(STEPS_PER_EPOCH_TRAIN)):
            batch_p_loss, batch_d_loss, batch_r_loss = pre_train_driver.train_step(inp)
            total_train_p_loss += batch_p_loss
            total_train_d_loss += batch_d_loss
            total_train_r_loss += batch_r_loss

        train_p_loss = total_train_p_loss / STEPS_PER_EPOCH_TRAIN
        train_d_loss = total_train_d_loss / STEPS_PER_EPOCH_TRAIN
        train_r_loss = total_train_r_loss / STEPS_PER_EPOCH_TRAIN

        if epoch % EVAL_INTERVAL == 0:

            # log train summary
            with train_summary_writer.as_default():
                tf.summary.scalar('pre_train_p_loss', train_p_loss, step=epoch)
                tf.summary.scalar('pre_train_d_loss', train_d_loss, step=epoch)
                tf.summary.scalar('pre_train_r_loss', train_r_loss, step=epoch)

            # perform validation
            valid_p_loss, valid_d_loss, valid_r_loss = pre_train_driver.test_step((x_valid, y_valid))

            # generate val. song attr.
            (g_p_out, g_d_out, g_r_out) = pre_train_driver.generate((x_valid, y_valid))
            g_p_out = one_hot(g_p_out)
            g_d_out = one_hot(g_d_out)
            g_r_out = one_hot(g_r_out)
            valid_g_out = (g_p_out, g_d_out, g_r_out)

            # infer val. song attr.
            y_valid_gen_attr = infer(valid_g_out, LE_PATHS, is_tune=True)

            # compute pitch, duration, rest & overall mmd score
            valid_p_mmd, valid_d_mmd, valid_r_mmd = compute_mmd_score(y_valid_dat_attr, y_valid_gen_attr)
            valid_o_mmd = valid_p_mmd + valid_d_mmd + valid_r_mmd

            # log validation summary
            with valid_summary_writer.as_default():
                tf.summary.scalar('pre_train_p_loss', valid_p_loss, step=epoch)
                tf.summary.scalar('pre_train_d_loss', valid_d_loss, step=epoch)
                tf.summary.scalar('pre_train_r_loss', valid_r_loss, step=epoch)

                tf.summary.scalar('pMMD', valid_p_mmd, step=epoch)
                tf.summary.scalar('dMMD', valid_d_mmd, step=epoch)
                tf.summary.scalar('rMMD', valid_r_mmd, step=epoch)
                tf.summary.scalar('oMMD', valid_o_mmd, step=epoch)

            # testing

            # generate test song attr.
            (g_p_out, g_d_out, g_r_out) = pre_train_driver.generate((x_test, y_test))
            g_p_out = one_hot(g_p_out)
            g_d_out = one_hot(g_d_out)
            g_r_out = one_hot(g_r_out)
            test_g_out = (g_p_out, g_d_out, g_r_out)

            # infer test song attr.
            y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=True)

            # compute self-bleu score
            test_self_bleu = compute_self_bleu_score(y_test_gen_attr, N_GRAMS)

            # compute pitch, duration, rest & overall mmd score
            test_p_mmd, test_d_mmd, test_r_mmd = compute_mmd_score(y_test_dat_attr, y_test_gen_attr)
            test_o_mmd = test_p_mmd + test_d_mmd + test_r_mmd

            # log test summary
            with test_summary_writer.as_default():
                for n_gram in N_GRAMS:
                    tf.summary.scalar(f'selfBLEU_{n_gram}', test_self_bleu[n_gram], step=epoch)

                tf.summary.scalar('pMMD', test_p_mmd, step=epoch)
                tf.summary.scalar('dMMD', test_d_mmd, step=epoch)
                tf.summary.scalar('rMMD', test_r_mmd, step=epoch)
                tf.summary.scalar('oMMD', test_o_mmd, step=epoch)

                # save test logs for current epoch

                for n_gram in N_GRAMS:
                    logs[f'selfBLEU_{n_gram}'] = test_self_bleu[n_gram]

                logs['pMMD'] = test_p_mmd
                logs['dMMD'] = test_d_mmd
                logs['rMMD'] = test_r_mmd
                logs['oMMD'] = test_o_mmd

            test_logs.append(logs)

            print ('Epoch {} Pitch Loss: Train: {:.4f} Validation: {:.4f}'.format(
                epoch+1, train_p_loss, valid_p_loss))

            print ('Epoch {} Duration Loss: Train: {:.4f} Validation: {:.4f}'.format(
                epoch+1, train_d_loss, valid_d_loss))

            print ('Epoch {} Rest Loss: Train: {:.4f} Validation: {:.4f}'.format(
                epoch+1, train_r_loss, valid_r_loss))

        # create a checkpoint
        pre_train_ckpt_save_path = pre_train_ckpt_manager.save()
        print('Saving pretrain checkpoint for epoch {} at {}'.format(
            epoch+1, pre_train_ckpt_save_path))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    ## Adversarial Training

    print('\n ***** Starting Adversarial Training ***** \n')

    # Temperature updates are done at step-level
    
    for epoch in range(ADV_TRAIN_EPOCHS):
        start = time.time()

        logs = {'seed' : SEED,
                'epoch': epoch+PRE_TRAIN_EPOCHS,
                'base' : 'rmc'}

        total_train_g_loss = 0
        total_train_d_loss = 0
        total_train_G_mi = 0
        total_train_D_mi = 0

        for step, inp in enumerate(train_dataset.take(STEPS_PER_EPOCH_TRAIN)):
            # update temperature
            adv_train_driver.update_temp(epoch, step)

            batch_g_loss, batch_d_loss, batch_g_out, batch_G_mi, batch_D_mi = adv_train_driver.train_step(inp)
            total_train_g_loss += batch_g_loss
            total_train_d_loss += batch_d_loss
            total_train_G_mi += batch_G_mi
            total_train_D_mi += batch_D_mi

            with train_summary_writer.as_default():
                tf.summary.scalar('temperature', tf.keras.backend.get_value(adv_train_driver.temp), 
                                  step=epoch*STEPS_PER_EPOCH_TRAIN+step)

        train_g_loss = total_train_g_loss/STEPS_PER_EPOCH_TRAIN
        train_d_loss = total_train_d_loss/STEPS_PER_EPOCH_TRAIN
        train_G_mi = total_train_G_mi/STEPS_PER_EPOCH_TRAIN
        train_D_mi = total_train_D_mi/STEPS_PER_EPOCH_TRAIN

        if epoch % EVAL_INTERVAL == 0:

            # log train summary
            with train_summary_writer.as_default():
                with train_summary_writer.as_default():
                    tf.summary.scalar('g_loss', train_g_loss, step=epoch)
                    tf.summary.scalar('d_loss', train_d_loss, step=epoch)
                    tf.summary.scalar('G_mi', train_G_mi, step=epoch)
                    tf.summary.scalar('D_mi', train_D_mi, step=epoch)

            # generate val. song attr.
            valid_g_loss, valid_d_loss, valid_g_out, valid_G_mi, valid_D_mi = adv_train_driver.test_step((x_valid, y_valid))

            # infer val. song attr.
            y_valid_gen_attr = infer(valid_g_out, LE_PATHS, is_tune=True)

            # compute pitch, duration, rest & overall mmd score
            valid_p_mmd, valid_d_mmd, valid_r_mmd = compute_mmd_score(y_valid_dat_attr, y_valid_gen_attr)
            valid_o_mmd = valid_p_mmd + valid_d_mmd + valid_r_mmd

            # log validation summary
            with valid_summary_writer.as_default():
                tf.summary.scalar('g_loss', valid_g_loss, step=epoch)
                tf.summary.scalar('d_loss', valid_d_loss, step=epoch)
                tf.summary.scalar('G_mi', valid_G_mi, step=epoch)
                tf.summary.scalar('D_mi', valid_D_mi, step=epoch)

                tf.summary.scalar('pMMD', valid_p_mmd, step=epoch+PRE_TRAIN_EPOCHS)
                tf.summary.scalar('dMMD', valid_d_mmd, step=epoch+PRE_TRAIN_EPOCHS)
                tf.summary.scalar('rMMD', valid_r_mmd, step=epoch+PRE_TRAIN_EPOCHS)
                tf.summary.scalar('oMMD', valid_o_mmd, step=epoch+PRE_TRAIN_EPOCHS)

            # testing

            # generate test song attr.
            test_g_out, _, _, _ = adv_train_driver.generate(x_test)

            # infer test song attr.
            y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=True)

            # compute self-bleu score
            test_self_bleu = compute_self_bleu_score(y_test_gen_attr, N_GRAMS)

            # compute pitch, duration, rest & overall mmd score
            test_p_mmd, test_d_mmd, test_r_mmd = compute_mmd_score(y_test_dat_attr, y_test_gen_attr)
            test_o_mmd = test_p_mmd + test_d_mmd + test_r_mmd

            # log test summary
            with test_summary_writer.as_default():
                for n_gram in N_GRAMS:
                    tf.summary.scalar(f'selfBLEU_{n_gram}', test_self_bleu[n_gram], step=epoch+PRE_TRAIN_EPOCHS)

                tf.summary.scalar('pMMD', test_p_mmd, step=epoch+PRE_TRAIN_EPOCHS)
                tf.summary.scalar('dMMD', test_d_mmd, step=epoch+PRE_TRAIN_EPOCHS)
                tf.summary.scalar('rMMD', test_r_mmd, step=epoch+PRE_TRAIN_EPOCHS)
                tf.summary.scalar('oMMD', test_o_mmd, step=epoch+PRE_TRAIN_EPOCHS)

                # save test logs for current epoch

                for n_gram in N_GRAMS:
                    logs[f'selfBLEU_{n_gram}'] = test_self_bleu[n_gram]

                logs['pMMD'] = test_p_mmd
                logs['dMMD'] = test_d_mmd
                logs['rMMD'] = test_r_mmd
                logs['oMMD'] = test_o_mmd

            test_logs.append(logs)

            print(
                'Epoch {} Train loss: G:{:.4f}, D:{:.4f}, Valid loss: G:{:.4f}, D:{:.4f}, G_mi: {:.4f}, D_mi: {:.4f}'.format(
                    epoch + 1, train_g_loss, train_d_loss, valid_g_loss, valid_d_loss, train_G_mi, train_D_mi))

        # create a checkpoint
        adv_train_ckpt_save_path = adv_train_ckpt_manager.save()

        print('Saving checkpoint for epoch {} at {}'.format(
          epoch+1, adv_train_ckpt_save_path))

        print('Temperature used for epoch {} : {}'.format(
          epoch+1, tf.keras.backend.get_value(adv_train_driver.temp)))

        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # save test logs as csv file

    pd.DataFrame.from_records(test_logs).to_csv(TEST_LOGS_FILENAME, index=False)

if __name__ == '__main__':
    
    settings = {'settings_file': 'settings'}
    settings = load_settings_from_file(settings)

    print("Settings: \n")
    for (k, v) in settings.items():
        print(v, '\t', k)
    
    locals().update(settings)
    
    print("================================================================ \n " )
    
    main()
    
    print("Training is complete.")
