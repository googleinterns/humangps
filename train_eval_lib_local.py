# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training library showing how to train the model using distributed strategy."""
import enum
import functools
from typing import Any, Callable, Dict, Optional, Union

from absl import logging
import gin.tf
import tensorflow as tf

import dataset_lib
from deep_visual_descriptor.model import geodesic_feature_network

Strategy = Union[tf.distribute.OneDeviceStrategy,
                 tf.distribute.MirroredStrategy,
                 tf.distribute.experimental.TPUStrategy]


@enum.unique
class TrainingMode(enum.Enum):
  """Training mode used for strategy."""
  CPU = 'cpu'
  GPU = 'gpu'
  TPU = 'tpu'


class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Build a one cycle learning rate schedule."""

  def __init__(
      self,
      initial_learning_rate: Union[float, tf.Tensor],
      maximal_learning_rate: Union[float, tf.Tensor],
      max_train_steps: Union[float, int, tf.Tensor],
      start_steps: Union[float, int, tf.Tensor] = 0,
      name: str = 'OneCycleLR',
  ):
    """Initialize a one cycle learning rate schedule.

    Args:
      initial_learning_rate: A scalar `float` or `int` or `Tensor`, The initial
        learning rate.
      maximal_learning_rate: A scalar `float` or `int` or `Tensor`, The initial
        learning rate.
      max_train_steps: A scalar `float` or `int` or `Tensor`, The maximal
        traning step for one cycle training. Must be positive.
      start_steps: A scalar `float` or `int` or `Tensor`. The start step, which
        will be added to the current step.
      name: String.  Optional name of the operation. Defaults to 'OneCycleLR'.
    """
    super().__init__()
    self._initial_learning_rate = initial_learning_rate
    self._maximal_learning_rate = maximal_learning_rate
    self._max_train_steps = max_train_steps
    self._peak_steps = max_train_steps / 5.0
    self._start_steps = start_steps
    self._name = name

  def __call__(self, step):
    """Return the learning rate for current step."""
    with tf.name_scope(self._name):
      step = tf.cast(step, tf.float32) - self._start_steps
      increasing_lr = tf.math.maximum(
          self._initial_learning_rate + step / self._peak_steps *
          (self._maximal_learning_rate - self._initial_learning_rate),
          self._initial_learning_rate)
      decreasing_lr = tf.math.maximum(
          self._maximal_learning_rate - (step - self._peak_steps) /
          (self._max_train_steps - self._peak_steps) *
          (self._maximal_learning_rate - self._initial_learning_rate),
          self._initial_learning_rate)
      learning_rate = tf.cond(
          tf.less(step, self._peak_steps), lambda: increasing_lr,
          lambda: decreasing_lr)
      return learning_rate

  def get_config(self):
    """Return a Dict of a configuration for LearningRateSchedule."""
    return {
        'initial_learning_rate': self._initial_learning_rate,
        'maximal_learning_rate': self._maximal_learning_rate,
        'max_train_steps': self._max_train_steps,
        'peak_steps': self._peak_steps,
        'start_steps': self._start_steps,
        'name': self._name,
    }


def get_strategy(training_mode: TrainingMode) -> Strategy:
  """Creates a distributed strategy."""
  strategy = None
  if training_mode == TrainingMode.CPU:
    strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
  elif training_mode == TrainingMode.GPU:
    strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
  else:
    raise ValueError('Unsupported distributed mode.')
  return strategy


def _concat_tensors(tensors: tf.Tensor) -> tf.Tensor:
  """Concatenate tensors of the different replicas."""
  return tf.concat(tf.nest.flatten(tensors, expand_composites=True), axis=0)


def _summary_writer(summaries_dict: Dict[str, Any],
                    step: Optional[int] = None) -> None:
  """Adds scalar and image summaries."""
  # Add scalar summaries.
  for key, scalars in summaries_dict['scalar_summaries'].items():
    tf.summary.scalar(key, scalars, step=step)
  # Add image summaries. Data for key 'images' has range [0, 1].
  for key, images in summaries_dict['image_summaries'].items():
    tf.summary.image(key, images, max_outputs=6, step=step)

  for key, histogram in summaries_dict['histogram_summaries'].items():
    tf.summary.histogram(key, histogram, step=step)


@tf.function
def _distributed_train_step(strategy: Strategy, batch: Dict[str, tf.Tensor],
                            model: tf.keras.Model,
                            optimizer: tf.keras.optimizers.Optimizer,
                            global_batch_size: int):
  """Distributed training step.

  Args:
    strategy: A Tensorflow distribution strategy.
    batch: A batch of training examples.
    model: The Keras model to train.
    optimizer: The Keras optimizer used to train the model.
    global_batch_size: The global batch size used to scale the loss.

  Returns:
    A dictionary of train step outputs.
  """

  def _train_step(batch):
    """Train for one step."""
    with tf.GradientTape() as tape:
      # Copy data to prevent the complaint about changing input.
      batch = batch.copy()
      training_loss, scalar_summaries, image_summaries = model.get_train_outputs(
          batch)
      loss = training_loss / global_batch_size
      loss = tf.debugging.check_numerics(loss, message='training loss is nan')

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    grads_summaries = {}
    return (scalar_summaries, image_summaries, grads_summaries)

  (scalar_summaries, image_summaries, grads_summaries) = strategy.run(
      _train_step, args=(batch,))

  loss = scalar_summaries['training_loss']
  loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
  loss = tf.debugging.check_numerics(
      loss, message='training loss is nan after strategy.reduce')

  for key in scalar_summaries:
    scalar_summaries[key] = strategy.reduce(
        tf.distribute.ReduceOp.MEAN, scalar_summaries[key], axis=None)

  for key in grads_summaries:
    scalar_summaries[key] = strategy.reduce(
        tf.distribute.ReduceOp.MEAN, grads_summaries[key], axis=None)

  for key in image_summaries:
    image_summaries[key] = _concat_tensors(image_summaries[key])

  histogram_summaries = {}
  return {
      'scalar_summaries': scalar_summaries,
      'image_summaries': image_summaries,
      'histogram_summaries': histogram_summaries,
  }


def train_loop(strategy: Strategy,
               train_set: tf.data.Dataset,
               create_model_fn: Callable[..., tf.keras.Model],
               create_optimizer_fn: Callable[...,
                                             tf.keras.optimizers.Optimizer],
               global_batch_size: int,
               base_folder: str,
               num_iterations: int,
               save_summaries_frequency: int = 100,
               save_checkpoint_frequency: int = 100,
               checkpoint_max_to_keep: int = 20,
               checkpoint_save_every_n_hours: float = 2.,
               timing_frequency: int = 100):
  """A Tensorflow 2 eager mode training loop.

  Args:
    strategy: A Tensorflow distributed strategy.
    train_set: A tf.data.Dataset to loop through for training.
    create_model_fn: A callable that returns a tf.keras.Model.
    create_optimizer_fn: A callable that returns a
      tf.keras.optimizers.Optimizer.
    global_batch_size: The global batch size, typically used to scale losses in
      distributed_train_step_fn.
    base_folder: A CNS path to where the summaries event files and checkpoints
      will be saved.
    num_iterations: An integer, the number of iterations to train for.
    save_summaries_frequency: The iteration frequency with which summaries are
      saved.
    save_checkpoint_frequency: The iteration frequency with which model
      checkpoints are saved.
    checkpoint_max_to_keep: The maximum number of checkpoints to keep.
    checkpoint_save_every_n_hours: The frequency in hours to keep checkpoints.
    timing_frequency: The iteration frequency with which to log timing.
  """
  summary_writer = tf.summary.create_file_writer(base_folder)
  summary_writer.set_as_default()

  train_set = strategy.experimental_distribute_dataset(train_set)
  with strategy.scope():
    logging.info('Building model ...')
    model = create_model_fn()
    optimizer = create_optimizer_fn()

  logging.info('Creating checkpoint ...')
  checkpoint = tf.train.Checkpoint(
      model=model,
      optimizer=optimizer,
      step=optimizer.iterations,
      epoch=tf.Variable(0, dtype=tf.int64, trainable=False),
      training_finished=tf.Variable(False, dtype=tf.bool, trainable=False))

  logging.info('Restoring old model (if exists) ...')
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=base_folder,
      max_to_keep=checkpoint_max_to_keep,
      keep_checkpoint_every_n_hours=checkpoint_save_every_n_hours)

  if checkpoint_manager.latest_checkpoint:
    with strategy.scope():
      checkpoint.restore(checkpoint_manager.latest_checkpoint)

  logging.info('Creating Timer ...')
  timer = tf.estimator.SecondOrStepTimer(every_steps=timing_frequency)
  timer.update_last_triggered_step(optimizer.iterations.numpy())

  logging.info('Training ...')

  while optimizer.iterations.numpy() < num_iterations:
    for i_batch, batch in enumerate(train_set):
      # Log epoch, total iterations and batch index.
      logging.info('epoch %d; iterations %d; i_batch %d',
                   checkpoint.epoch.numpy(), optimizer.iterations.numpy(),
                   i_batch)

      # Break if the number of iterations exceeds the max.
      if optimizer.iterations.numpy() >= num_iterations:
        break

      # Compute distributed step outputs.
      distributed_step_outputs = _distributed_train_step(
          strategy, batch, model, optimizer, global_batch_size)

      # Save checkpoint.
      if optimizer.iterations.numpy() % save_checkpoint_frequency == 0:
        checkpoint_manager.save(checkpoint_number=optimizer.iterations.numpy())

      # Write summaries.
      if optimizer.iterations.numpy() % save_summaries_frequency == 0:
        tf.summary.experimental.set_step(step=optimizer.iterations.numpy())
        _summary_writer(distributed_step_outputs)

      # Log steps/sec.
      if timer.should_trigger_for_step(optimizer.iterations.numpy()):
        elapsed_time, elapsed_steps = timer.update_last_triggered_step(
            optimizer.iterations.numpy())
        if elapsed_time is not None:
          steps_per_second = elapsed_steps / elapsed_time
          tf.summary.scalar(
              'steps/sec', steps_per_second, step=optimizer.iterations)
          tf.summary.scalar(
              'learn_rate',
              optimizer.learning_rate(optimizer.iterations),
              step=optimizer.iterations)

    # Increment epoch.
    checkpoint.epoch.assign_add(1)

  # Assign training_finished variable to True after training is finished and
  # save the last checkpoint.
  checkpoint.training_finished.assign(True)
  checkpoint_manager.save(checkpoint_number=optimizer.iterations.numpy())


@gin.configurable
def get_training_elements(
    model_component: str,
    model_hparams: Dict[str, Any]) -> Callable[..., tf.keras.Model]:
  """Get model architecture."""

  if model_component == 'GeoFeatureNet':
    create_model_fn = functools.partial(geodesic_feature_network.GeoFeatureNet,
                                        model_hparams)
  else:
    raise ValueError('Unknown model_component: %s' % model_component)

  return create_model_fn


def get_training_optimizer(
    lr_params: Dict[str, Any]
) -> Callable[..., tf.keras.optimizers.schedules.LearningRateSchedule]:
  """Get training optimizer."""
  if 'piecewise_lr' in lr_params:
    learning_rate = lr_params['piecewise_lr']['learning_rate']
    start_steps = lr_params['piecewise_lr']['start_steps']
    boundaries = [200000 * i + start_steps for i in range(2, 7)]
    values = [learning_rate * pow(1.25, -i) for i in range(6)]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values, name=None)
    create_optimizer_fn = functools.partial(
        tf.keras.optimizers.Adam,
        learning_rate=lr_schedule,
        beta_1=0.5,
        beta_2=0.999)

  elif 'one_cycle_lr' in lr_params:
    lr_schedule = OneCycleLR(
        initial_learning_rate=lr_params['one_cycle_lr']['learning_rate'] / 25.0,
        maximal_learning_rate=lr_params['one_cycle_lr']['learning_rate'],
        max_train_steps=lr_params['one_cycle_lr']['max_train_steps'],
        start_steps=lr_params['one_cycle_lr']['start_steps'],
    )
    create_optimizer_fn = functools.partial(
        tf.keras.optimizers.Adam,
        learning_rate=lr_schedule,
        beta_1=0.5,
        beta_2=0.999)
  else:
    raise Exception('Unknown Learning rate schedule')
  return create_optimizer_fn


@gin.configurable
def train_pipeline(training_mode: str,
                   base_folder: str,
                   dataset_params: Dict[str, Any],
                   lr_params: Dict[str, Any],
                   batch_size: int,
                   n_iterations: int,
                   save_every_n_batches: int = 100,
                   time_every_n_steps: int = 100):
  """A training function that is strategy agnostic.

  Args:
    training_mode: Distributed strategy approach, one from 'cpu', 'gpu', 'tpu'.
    base_folder: A CNS path to where the summaries event files and checkpoints
      will be saved.
    dataset_params: Dict to describe dataset parameters.
    lr_params: Dict to describe learning rate schedule parameters.
    batch_size: An integer, the batch size.
    n_iterations: An integer, the number of iterations to train for.
    save_every_n_batches: An integer, save n_batches / save_every_n_batches.
    time_every_n_steps: An integer, report timing this often.
  """
  logging.info('Loading training data ...')

  # Sets model configuration parameters
  dataset_params['batch_size'] = batch_size
  train_set = dataset_lib.load_dataset(dataset_params)

  create_model_fn = get_training_elements(
      model_component=gin.REQUIRED, model_hparams=gin.REQUIRED)

  create_optimizer_fn = get_training_optimizer(lr_params)

  train_loop(
      strategy=get_strategy(TrainingMode(training_mode)),
      train_set=train_set,
      create_model_fn=create_model_fn,
      create_optimizer_fn=create_optimizer_fn,
      global_batch_size=batch_size,
      base_folder=base_folder,
      num_iterations=n_iterations,
      save_summaries_frequency=time_every_n_steps,
      save_checkpoint_frequency=save_every_n_batches,
      timing_frequency=time_every_n_steps)


@tf.function
def _distributed_eval_step(strategy: Strategy, batch: Dict[str, tf.Tensor],
                           model: tf.keras.Model) -> Dict[str, Any]:
  """Distributed eval step.

  Args:
    strategy: A Tensorflow distribution strategy.
    batch: A batch of training examples.
    model: The Keras model to evaluate.

  Returns:
    A dictionary holding summaries.
  """

  def _eval_step(batch: Dict[str, tf.Tensor]):
    """Eval for one step."""
    # Copy data to prevent the complaint about changing input.
    batch = batch.copy()
    _, scalar_summaries, image_summaries = model.get_eval_outputs(batch)

    return (scalar_summaries, image_summaries)

  (scalar_summaries, image_summaries) = strategy.run(_eval_step, args=(batch,))
  for key in scalar_summaries:
    scalar_summaries[key] = strategy.reduce(
        tf.distribute.ReduceOp.MEAN, scalar_summaries[key], axis=None)

  for key in image_summaries:
    image_summaries[key] = _concat_tensors(image_summaries[key])

  histogram_summaries = {}

  return {
      'scalar_summaries': scalar_summaries,
      'image_summaries': image_summaries,
      'histogram_summaries': histogram_summaries
  }


@gin.configurable
def eval_pipeline(eval_mode: str, dataset_params: Dict[str, Any],
                  train_base_folder: str, eval_base_folder: str,
                  batch_size: int, eval_name: str):
  """A eval function that is strategy agnostic.

  Args:
    eval_mode: Distributed strategy approach, one from 'cpu', 'gpu', 'tpu'.
    dataset_params: Dictionary of files that make up the dataset for
      experiments.
    train_base_folder: A CNS path to where the training checkpoints will be
      loaded.
    eval_base_folder: A CNS path to where the evaluation summaries event files
      will be saved.
    batch_size: An integer, the batch size.
    eval_name: The experiment name.
  """
  strategy = get_strategy(TrainingMode(eval_mode))

  logging.info('Creating summaries ...')
  summary_writer = tf.summary.create_file_writer(eval_base_folder)
  summary_writer.set_as_default()

  logging.info('Loading testing data ...')

  # Sets model configuration parameters
  dataset_params = dataset_params[eval_name]
  dataset_params['batch_size'] = batch_size
  test_set = dataset_lib.load_dataset(dataset_params)

  test_set = strategy.experimental_distribute_dataset(test_set)

  create_model_fn = get_training_elements(
      model_component=gin.REQUIRED, model_hparams=gin.REQUIRED)

  with strategy.scope():
    logging.info('Building model ...')
    model = create_model_fn()
    if hasattr(model, 'path_drop_probabilities'):
      if model.path_drop_probabilities[
          0] != 1.0 or model.path_drop_probabilities[1] != 1.0:
        # When evaluate inter-subject dataset, set the weight of flow path to 0.
        if eval_name == 'eval_optical_flow_inter':
          model.path_drop_probabilities = [0.0, 1.0]
        else:
          model.path_drop_probabilities = [1.0, 1.0]

  checkpoint = tf.train.Checkpoint(
      model=model,
      step=tf.Variable(-1, dtype=tf.int64),
      training_finished=tf.Variable(False, dtype=tf.bool),
  )

  for checkpoint_path in tf.train.checkpoints_iterator(
      train_base_folder,
      min_interval_secs=10,
      timeout=10,
      timeout_fn=lambda: checkpoint.training_finished):

    try:
      status = checkpoint.restore(checkpoint_path)
      status.assert_existing_objects_matched()
      status.expect_partial()
    except (tf.errors.NotFoundError, AssertionError) as err:
      logging.info('Failed to restore checkpoint from %s. Error:\n%s',
                   checkpoint_path, err)
      continue

    logging.info('Restoring checkpoint %s @ step %d.', checkpoint_path,
                 checkpoint.step)

    logging.info('Evaluating ...')

    eval_record = {}
    eval_batch_scalar = {}

    for i_batch, batch in enumerate(test_set):
      logging.info('i_batch %d', i_batch)
      distributed_step_outputs = _distributed_eval_step(strategy, batch, model)
      if i_batch == 0:
        eval_record['image_summaries'] = distributed_step_outputs[
            'image_summaries'].copy()
      for key, scalar in distributed_step_outputs['scalar_summaries'].items():
        if key in eval_batch_scalar:
          eval_batch_scalar[key].append(scalar)
        else:
          eval_batch_scalar[key] = [scalar]

    eval_record['scalar_summaries'] = {}
    for key, record in eval_batch_scalar.items():
      eval_record['scalar_summaries'][key] = tf.reduce_mean(record)

    eval_record['histogram_summaries'] = distributed_step_outputs[
        'histogram_summaries']

    for key in eval_record['scalar_summaries']:
      print('%s: %f' % (key, eval_record['scalar_summaries'][key]))

    _summary_writer(eval_record, step=checkpoint.step)


@gin.configurable
def inference_pipeline(eval_mode: str, dataset_params: Dict[str, Any],
                  checkpoint_path: str, batch_size: int, eval_name: str):
  """A eval function that is strategy agnostic.

  Args:
    eval_mode: Distributed strategy approach, one from 'cpu', 'gpu'.
    dataset_params: Dictionary of files that make up the dataset for
      experiments.
    checkpoint_path: A path to where the training checkpoints will be
      loaded.
    batch_size: An integer, the batch size.
    eval_name: The experiment name.
  """
  strategy = get_strategy(TrainingMode(eval_mode))

  logging.info('Loading testing data ...')

  # Sets model configuration parameters
  dataset_params = dataset_params[eval_name]
  dataset_params['batch_size'] = batch_size
  test_set = dataset_lib.load_dataset(dataset_params)

  test_set = strategy.experimental_distribute_dataset(test_set)

  create_model_fn = get_training_elements(
      model_component=gin.REQUIRED, model_hparams=gin.REQUIRED)

  with strategy.scope():
    logging.info('Building model ...')
    model = create_model_fn()

  checkpoint = tf.train.Checkpoint(
      model=model,
      step=tf.Variable(-1, dtype=tf.int64),
      training_finished=tf.Variable(False, dtype=tf.bool),
  )

  status = checkpoint.restore(checkpoint_path)
  status.assert_existing_objects_matched()
  status.expect_partial()

  logging.info('Restoring checkpoint %s @ step %d.', checkpoint_path,
               checkpoint.step)

  logging.info('Evaluating ...')

  eval_record = {}
  eval_batch_scalar = {}

  for i_batch, batch in enumerate(test_set):
    logging.info('i_batch %d', i_batch)
    distributed_step_outputs = _distributed_eval_step(strategy, batch, model)

    for key, scalar in distributed_step_outputs['scalar_summaries'].items():
      if key in eval_batch_scalar:
        eval_batch_scalar[key].append(scalar)
      else:
        eval_batch_scalar[key] = [scalar]
    if i_batch > 20:
      break

  eval_record['scalar_summaries'] = {}
  for key, record in eval_batch_scalar.items():
    eval_record['scalar_summaries'][key] = tf.reduce_mean(record)

  for key in eval_record['scalar_summaries']:
    print('%s: %f' % (key, eval_record['scalar_summaries'][key]))

