
import tensorflow as tf

import numpy as np

from sklearn.preprocessing import LabelEncoder
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import pickle
import json

import midi_statistics
import pretty_midi
import mmd

"""# Utils"""

def create_categorical_2d_encoder(x2d, le_path):
    """
    :param x2d: an attribute of a note [pitch, duration or rest]
    :shape x2d: [None, SONG_LENGTH]
    """
    le = LabelEncoder()
    x1d = x2d.ravel()
    le.fit(x1d)

    classes = le.classes_
    print(len(classes))
    pickle.dump(le, open(le_path, "wb"))

    return len(classes)

def to_categorical_2d(x2d, le_path):
    """
    :param x2d: an attribute of a note [pitch, duration or rest]
    :shape x2d: [None, SONG_LENGTH]

    :return y3d: one hot representation of note attribute
    :shape  y3d: [None, SONG_LENGTH, NUM_[.]_TOKENS]
    """
    le = pickle.load(open(le_path, "rb"))
    x1d = x2d.ravel()
    x1d = le.transform(x1d)

    n_classes = len(le.classes_)
    y1d = tf.keras.utils.to_categorical(x1d, n_classes)

    n_cls = y1d.shape[1]
    n_rows, n_cols = x2d.shape
    y3d = y1d.reshape( [ n_rows, n_cols, n_cls ] )

    return y3d

def from_categorical_2d(x2d, le_path):
    """
    :param x2d: an attribute of a note [pitch, duration or rest]
    :shape x2d: [None, SONG_LENGTH]
    """
    x2d = np.asarray(x2d) # ensure x2d is a ndarray
    le = pickle.load(open(le_path, "rb"))

    nrows = x2d.shape[0]
    ncols = x2d.shape[1]

    x1d  = x2d.ravel()
    x1d  = le.inverse_transform(x1d) 
    x2d = np.reshape(x1d, (nrows, ncols))

    return x2d

def load_data(path, le_paths, song_length, num_song_features, num_meta_features, 
              is_unique=False, convert_to_tensor=False):
  
    data = np.load(path).astype(np.float32)

    if is_unique: 
        data = np.unique(data, axis=0)

    x = data[:, song_length * num_song_features:]           # 400 cols : 60 => 460
    x = np.reshape(x, (-1, song_length, num_meta_features)) # 400 col => 20 x 20 cols

    y = data[:, :song_length * num_song_features]           # 60 cols
    y = np.reshape(y, (-1, song_length, num_song_features)) # 60 = 20 x 3: 1 pitch, 2 duration, 3 rest

    y_p = y[:, :, 0]
    y_d = y[:, :, 1]
    y_r = y[:, :, 2]

    y_p_ohe = to_categorical_2d(y_p, le_paths[0])
    y_d_ohe = to_categorical_2d(y_d, le_paths[1])
    y_r_ohe = to_categorical_2d(y_r, le_paths[2])

    if convert_to_tensor: 
        x = tf.convert_to_tensor(x)

        y_p_ohe = tf.convert_to_tensor(y_p_ohe)
        y_d_ohe = tf.convert_to_tensor(y_d_ohe)
        y_r_ohe = tf.convert_to_tensor(y_r_ohe)

    return x, (y_p, y_d, y_r), (y_p_ohe, y_d_ohe, y_r_ohe)

def tune_pitch(y_attr_p):
    """
    :param y_attr_p : pitch attribute of a musical note.
    """
    y_attr_p = np.expand_dims(y_attr_p, axis=-1) # [None, SONG_LENGTH, 1]
    y_attr_p = list(map(midi_statistics.tune_song, y_attr_p)) # [None, SONG_LENGTH, 1]
    return np.squeeze(y_attr_p, axis=-1) # [None, SONG_LENGTH]

def infer(y, le_paths, is_tune=False):
    """
    :param y: (pitch, duration, rest) i.e. attributes of a note
    :type  y: tuple
    :shape y[.]: [None, SONG_LENGTH]
    """
    y_attr_p, y_attr_d, y_attr_r = y

    y_attr_p = from_categorical_2d(y_attr_p, le_paths[0]) # [None, SONG_LENGTH]
    y_attr_d = from_categorical_2d(y_attr_d, le_paths[1]) # [None, SONG_LENGTH]
    y_attr_r = from_categorical_2d(y_attr_r, le_paths[2]) # [None, SONG_LENGTH]

    if is_tune: 
        y_attr_p = tune_pitch(y_attr_p)

    return y_attr_p, y_attr_d, y_attr_r

def gather_song_attr(y_attr, shape):
    """
    :param y: (pitch, duration, rest) i.e. attributes of a note
    :type  y: tuple
    :shape y[.]: [None, SONG_LENGTH]

    :param shape: [None, SONG_LENGTH, NUM_SONG_FEATURES]
    """
    y_attr_p, y_attr_d, y_attr_r = y_attr
    songs = np.zeros(shape)
    for i in range(shape[0]):
        for j, (p, d, r) in enumerate(
            zip(y_attr_p[i], y_attr_d[i], y_attr_r[i])):
            songs[i, j, 0] = p
            songs[i, j, 1] = d
            songs[i, j, 2] = r

    return songs

def gather_stats(y, shape):
    model_stats = {
        'songlength_tot':0,
        'stats_scale_tot':0,
        'stats_repetitions_2_tot':0,
        'stats_repetitions_3_tot':0,
        'stats_span_tot':0,
        'stats_unique_tones_tot':0,
        'stats_avg_rest_tot':0,
        'num_of_null_rest_tot':0,
        'best_scale_score':0,
        'best_repetitions_2':0,
        'best_repetitions_3':0,
        'num_perfect_scale':0,
        'num_good_songs':0}

    songs = gather_song_attr(y, shape)

    # gather stats
    for song in songs:
        song = song.tolist() # midi_pattern is list of array/list [array[p, d, r],...]
        stats = midi_statistics.get_all_stats(song)
        model_stats['songlength_tot'] += stats['songlength']
        model_stats['stats_scale_tot'] += stats['scale_score']
        model_stats['stats_repetitions_2_tot'] += float(stats['repetitions_2'])
        model_stats['stats_repetitions_3_tot'] += float(stats['repetitions_3'])
        model_stats['stats_unique_tones_tot'] += float(stats['tones_unique'])
        model_stats['stats_span_tot'] += stats['tone_span']
        model_stats['stats_avg_rest_tot'] += stats['average_rest']
        model_stats['num_of_null_rest_tot'] += stats['num_null_rest']
        model_stats['best_scale_score'] = max(stats['scale_score'], model_stats['best_scale_score'])
        model_stats['best_repetitions_2'] = max(stats['repetitions_2'], model_stats['best_repetitions_2'])
        model_stats['best_repetitions_3'] = max(stats['repetitions_3'], model_stats['best_repetitions_3'])

        if (stats['scale_score'] == 1.0 and stats['tones_unique'] > 3 and 
            stats['tone_span'] > 4 and stats['num_null_rest'] > 8 and 
            stats['tone_span'] < 13 and stats['repetitions_2'] > 4):
            model_stats['num_good_songs'] += 1

    return model_stats

def get_mean_stats(model_stats, num_songs):
    mean_model_stats = {}

    mean_model_stats['stats_scale'] = model_stats['stats_scale_tot'] / num_songs
    mean_model_stats['stats_repetitions_2'] = model_stats['stats_repetitions_2_tot'] / num_songs
    mean_model_stats['stats_repetitions_3'] = model_stats['stats_repetitions_3_tot'] / num_songs
    mean_model_stats['stats_span'] = model_stats['stats_span_tot'] / num_songs
    mean_model_stats['stats_unique_tones'] = model_stats['stats_unique_tones_tot'] / num_songs
    mean_model_stats['stats_avg_rest'] = model_stats['stats_avg_rest_tot'] / num_songs
    mean_model_stats['num_of_null_rest'] = model_stats['num_of_null_rest_tot'] / num_songs
    mean_model_stats['songlength'] = model_stats['songlength_tot'] / num_songs

    return mean_model_stats

def compute_mmd_score(y_dat_attr, y_gen_attr):
    """
    :param y_dat_attr: ground truth note attributes
    :type  y_dat_attr: tuple
    :shape y_dat_attr[.]: [None, SONG_LENGTH]

    :param y_gen_attr: generated note attributes
    :type  y_dat_attr: tuple
    :shape y_dat_attr[.]: [None, SONG_LENGTH]
    """
    y_dat_attr_p, y_dat_attr_d, y_dat_attr_r = y_dat_attr
    y_gen_attr_p, y_gen_attr_d, y_gen_attr_r = y_gen_attr

    mmd_p = mmd.Compute_MMD(y_gen_attr_p, y_dat_attr_p)
    mmd_d = mmd.Compute_MMD(y_gen_attr_d, y_dat_attr_d)
    mmd_r = mmd.Compute_MMD(y_gen_attr_r, y_dat_attr_r)

    return mmd_p, mmd_d, mmd_r

def get_note_sequence(y_attr):
    """
    This function computes note sequence using pitch, duration and rest sequences.
    :type y_attr: tuple
    :shape y_attr[.]: [None, SONG_LENGTH]

    :rtype: ndarray
    :rshape: [None, SONG_LENGTH]
    """
    y_p_attr, y_d_attr, y_r_attr = y_attr
    y_n = np.zeros(y_p_attr.shape, dtype=object)
    for i in range(y_n.shape[0]):
        for j in range(y_n.shape[1]):
            y_n[i, j] = (str(y_p_attr[i, j]) + 
                         str(y_d_attr[i, j]) + 
                         str(y_r_attr[i, j]))
    return y_n

def compute_corpus_bleu(hyp, ref, n_grams):
    """
    :param hyp: hypothesis
    :type  hyp: list(list(str))

    :param ref: list of references
    :type  ref: list(list(list(str)))
    """
    res = {}
    for n_gram in n_grams:
        weights = tuple((1. / n_gram for _ in range(n_gram)))
        res[n_gram] = corpus_bleu(ref, hyp, weights=weights, 
                                  smoothing_function=SmoothingFunction().method1)
    return res

def compute_self_bleu(x, n_grams, sample_size):
    """
    Compute self-BLEU score as a metric of sample diversity.
    High self-BLEU => Less sample diversity => High chance of mode collapse
    Low  self-BLEU => High sample diversity => Less chance of mode collapse
    :param x: generated output or hypothesis
    :type  x: ndarray
    :ndim  x: 2

    :param sample_size: self-BLEU score remains unchanged when sample_size >= 200
    """
    hyp = x[:sample_size].tolist() # list(list(str))
    ref = [np.delete(x, [i], axis=0).tolist() for i in range(sample_size)]
    return compute_corpus_bleu(hyp, ref, n_grams)

def compute_self_bleu_score(y_gen_attr, n_grams, sample_size=200):
    y_gen_n = get_note_sequence(y_gen_attr)
    return compute_self_bleu(y_gen_n, n_grams, sample_size)

def create_linear_initializer(input_size):
    """Returns a default initializer for weights of a linear module."""
    stddev = 1 / tf.sqrt(input_size * 1.0)
    return tf.keras.initializers.TruncatedNormal(stddev=stddev)

def load_settings_from_file(settings):
    """
    Handle loading settings from a JSON file, filling in missing settings from
    the command line defaults, but otherwise overwriting them.
    """
    settings_path = './settings/' + settings['settings_file'] + '.txt'
    print('Loading settings from', settings_path)
    settings_loaded = json.load(open(settings_path, 'r'))
    settings.update(settings_loaded)
    return settings

def create_midi_pattern_from_discretized_data(discretized_sample):
    new_midi = pretty_midi.PrettyMIDI()
    voice = pretty_midi.Instrument(1)  # It's here to change the used instruments !
    tempo = 120
    ActualTime = 0  # Time since the beginning of the song, in seconds
    for i in range(0,len(discretized_sample)):
        length = discretized_sample[i][1] * 60 / tempo  # Conversion Duration to Time
        if i < len(discretized_sample):
            gap = discretized_sample[i][2] * 60 / tempo
        else:
            gap = 0  # The Last element doesn't have a gap
        note = pretty_midi.Note(velocity=100, pitch=int(discretized_sample[i][0]), start=ActualTime,
                                end=ActualTime + length)
        voice.notes.append(note)
        ActualTime += length + gap  # Update of the time

    new_midi.instruments.append(voice)

    return new_midi

def one_hot(attrs):
    song_out = []
    for attr in attrs:
        attr_out = []
        for t in attr:
            g_t = tf.argmax(t, 0)
            attr_out.append(g_t)
        song_out.append(attr_out)
    out = tf.concat(np.array(song_out), axis=1)
    return out
