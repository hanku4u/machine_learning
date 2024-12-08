import tensorflow as tf
from helper_functions import cd_targets

@tf.keras.utils.register_keras_serializable(package='Custom')
class WeightedMSELoss(tf.keras.losses.Loss):
    def __init__(self, cd_targets, higher_weight=5.0, base_weight=1.0, name='weighted_mse_loss'):
        super().__init__(name=name)
        
        # store config parameters
        self.cd_targets = cd_targets
        self.higher_weight = higher_weight
        self.base_weight = base_weight

        # upper and lower limits as tensors
        upper_limits = []
        lower_limits = []
        for cd in cd_targets.keys():
            upper_cd, lower_cd, _ = cd_targets[cd]
            upper_limits.append(upper_cd)
            lower_limits.append(lower_cd)

        self.upper_limits = tf.constant(upper_limits, dtype=tf.float32)
        self.lower_limits = tf.constant(lower_limits, dtype=tf.float32)

    def call(self, y_true, y_pred):
        # compute standard MSE loss for each sample
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)

        # find out of spec samples
        out_of_spec_upper = tf.greater(y_true, self.upper_limits)
        out_of_spec_lower = tf.less(y_true, self.lower_limits)
        out_of_spec = tf.logical_or(out_of_spec_upper, out_of_spec_lower)

        # convert boolean mask to float
        out_of_spec = tf.cast(out_of_spec, tf.float32)

        # compute sample weights
        sample_weights = self.base_weight + (self.higher_weight - self.base_weight) * out_of_spec
        # if any CD is out of spec, consider the sample out of spec
        sample_weights = tf.reduce_max(sample_weights, axis=1)

        # Weighted MSE
        weighted_mse = mse * sample_weights
        return tf.reduce_mean(weighted_mse)

    def get_config(self):
        ''' Return a dictionary of config parameters so that the loss function can be recreated when loading the model '''
        return {
            'cd_targets': self.cd_targets,
            'higher_weight': self.higher_weight,
            'base_weight': self.base_weight,
            'name': self.name
        }

    @classmethod
    def from_config(cls, config):
        # Optionally, you can override from_config if needed.
        # In this case, the default implementation is usually sufficient.
        return cls(**config)
    
custom_loss_fn = WeightedMSELoss(cd_targets=cd_targets, higher_weight=20.0, base_weight=1.0)