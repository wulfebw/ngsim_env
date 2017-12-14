
import io
import numpy as np
import tensorflow as tf

import sys
import matplotlib
backend = 'Agg' if sys.platform == 'linux' else 'TkAgg'
matplotlib.use(backend)
import matplotlib.pyplot as plt

from hgail.misc.validator import Validator

class AutoValidator(Validator):

    def _summarize_env_infos(self, env_infos):
        summaries = []

        # means 
        mean_keys = ['rmse', 'is_colliding']
        for key in mean_keys:
            mean = np.mean(env_infos[key])
            tag = 'validation/mean_{}'.format(key)
            summaries += [tf.Summary.Value(tag=tag, simple_value=mean)]

        # hist
        hist_keys = ['rmse']
        for key in hist_keys:
            plt.hist(env_infos[key], 50)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_sum = tf.Summary.Image(encoded_image_string=buf.getvalue())
            tag = 'validation/hist_{}'.format(key)
            summaries += [tf.Summary.Value(tag=tag, image=img_sum)]
            plt.close()

        return summaries

    def validate(self, itr, objs):
        summaries = []

        if 'samples_data' in objs.keys():
            env_infos = objs['samples_data']['env_infos']
            summaries += self._summarize_env_infos(env_infos)

        self.write_summaries(itr, summaries)
        