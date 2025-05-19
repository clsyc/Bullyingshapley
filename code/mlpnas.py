"""
Controller-based neural architecture searc (NAS) model whose main objective is sampling multi-layer perceptrons (MLPs). This can be seen as a toy problem, since much more complex models should be sampled to tackle real-world problems, for example CNNs, and those would need a much deeper thinking on how to structure the search space to describe every single convolutional layer. This NAS system is based on a LSTM model whose main objective is to sample a probability distribution for every possible layer accounted by the search space, and the extract the next predicted layer through a random choice based on this distribution. At every epoch the controller calls the MLP generator for a number of times equal to desired number of architectures per-epoch to sample the architectures. The layers of these architectures are sampled in terms of unique indexes compliant to the vocabulary, that map to specific (N. of neurons, activation function) pair. When all of the layers for a single architecture have been sampled, the whole architecture is decoded back to actual Keras layers. After the sampling phase each architecture is trained on its dataset of choice for approximately 10 epochs with an early stopping of 3. This training procedure can actually take quite some time depending on the complexity of the dataset, so this must be one of the deciding factors of both the number of sampling epochs the controller performs and the number of architectures being sampled per-epoch.

Design choices
--------------
Of course more LSTM layers could have been implied to sample the probability distributions, but we didn't test it ouy. Also, a few questions which may actually sound more appealing are what the true role of the NAS' controller is and how the LSTM itself would actually be trained.
Its main role is to assure that the sequence being sampled by the MLP generator are valid and that no repeating ones are being sampled to avoid wasting important computational resources on regions of the search space which have already been explored. But how well the controller traverses the search space depends on how well it is trained to follow the more optimal directions. How so? In order to that we can see the controller as a LSTM that's being iteratively trained on the sequences that it generates. This starts with a controller that generates sequences without any knowledge of what an architecture that performs well looks like. After the first few sequences have been generated, trained and evaluated they will form a dataset the controller is fed with. In essence, after every controller epoch, a new dataset is created for the controller to learn from with the architectures sampled ever since the first epoch and this should make it learn how to discriminate within its hiden states between architectures that perform well from those that do not. About the actual training itself, since the controller must be aware of the architectures validation accuracies, while performing as an agent regulating what happens on its actions (which architectures to sample) and its states (its knowledge of the search space), the loss implements the REINFORCE algorithm, which is a typical algorithm in reinforce learning that implements a Monte-Carlo variant of a stochastic policy gradient. Its objective then is to learn a policy that maximizes the cumulative future reward score R, computed on the (Action, state) pair the controller was in, where the policy is defined as a probability distribution of actions where actions with a higher expected reward have a higher probability for a given observed state.

Policy gradient
-------------------
Given an objective function J, defined as: J(theta) = E[sum_{t = 1}^T r_t], where theta represents the policy parameter and r_t the reward at the t-th time step computed from the reward function taken into account, as R(s_t, a_t), the algorithm performs a gradient descent following the partial derivative of J w.r.t. theta looking for the optimal trajectory.

theta <-- theta + frac{d}{d theta} J(theta)

REINFORCE algorithm
-------------------
J(theta) = alpha*gamma^t G*[Nabla_theta ln pi(a_t|s_t, theta)]

For every epoch:
    Generate an episode, (s_1, a_1, r_1), ..., (s_T, a_T, r_T), following the current policy pi(.|., theta)
    For every step of the episode t = 1, ..., T:
        Compute the discounted cumulative reward G(t)
        theta <-- theta + alpha*gamma^t G*[Nabla_theta ln pi(a_t|s_t, theta)]

where alpha is the learning rate, gamma the discount factor and pi(a_t|s_t, theta) the probability of the occurence of (s_t, a_t) given the current trajectory followed by the NAS controller. It is evident here that the REINFORCE algorithm tries to maximize an objective J made of the product of the cumulative future rewarded G(t) discounted w.r.t. a baseline (0.5, in our case, that basically makes it so that about half of the actions, i.e. sequences of architectures, will end up being positive, hence good, and half negative, hence bad) with log-probabilites of actions on the policy ln pi(A|S, theta), where A = [a_1, ..., a_T], regulated by a learning rate, alpha. Indeed, the learning rate is a key factor in traversing appropriate regions of the search space, because the NAS model could very easily get stuck in regions where no good architectures can be found for the problem at hand. Hence, the optimizer (Adam, in our case) needs extra-care.

Since in REINFORCE the expectation of the sample gradient is equal to the actual gradient (TODO: Insert equation), it reflects a good theoretical convergence property, albeit being a Monte-Carlo-based method it may suffer from high variance.

Accuracy predictor
------------------
Our implementation opens up for the possibility to introduce an accuracy predictor, a network parallel to the LSTM itself, which turns the model into an adversarial model by accounting for an optimization that doesn't focues solely on the above REINFORCE loss function, but also on how well the accuracy predictor is becoming in predicting the goodness of the sampled architectures on their task.
Indeed the predictor is implemented by using a single dense layer which will share weights with the LSTM layer of the sequence layer, fed with the true validation accuracies reported by the previously sampled architectures at time t, thus letting it construct an internal representation of the architectures that allows it to understand the properties that characterize a good architecture as opposed to a bad one, without the need to train them for those 10 epochs, as a proxy to their validation accuracies. The controller, on the other hand, will try to navigate the search space in a way that also allows it to generate architectures not easily predictable by the predictor.

Despite the adversion of the predictor may lead to architectures with lower validation accuracies on some tasks than what they'd have without, its usage is still desired to help sampling architectures that generalize better.
"""

import pickle
import os
import numpy as np
# from sklearn.base import accuracy_score
import keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers
# Keras utilities (v. 2.4.0)
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# Keras callbacks for training
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score, accuracy_score
from tensorflow.keras.utils import to_categorical

# Local modules imports
# Note: Refer to constants.py for their meaning
from constants import CONTROLLER_SAMPLING_EPOCHS, SAMPLES_PER_CONTROLLER_EPOCH, CONTROLLER_TRAINING_EPOCHS, ARCHITECTURE_TRAINING_EPOCHS, CONTROLLER_LOSS_ALPHA, MAX_ARCHITECTURE_LENGTH, BEST_MODEL_PATH

from controller import Controller
from mlp_generator import MLPGenerator
from utils import clean_log, unison_shuffled_copies, sort_search_data

from tqdm import tqdm
import random
import pandas as pd
import matplotlib.pyplot as plt
import shap


class MLPNAS(Controller):
    def __init__(self, X, y, x_val, y_val, num_classes, task_name='default', reward_weights=None):
        """
        Parameters
        ----------
        task_name : str ['default']
            Name of the current task
            (Display purposes)
        """
        # 定义奖励权重
        if reward_weights is None:
            # 默认权重，可以根据您的需求调整
            self.reward_weights = {
                "accuracy": 0.6,
                "f1_macro": 0.2,
                "recall_macro": 0.2
            }
        else:
            self.reward_weights = reward_weights
        
        print(f"Using reward weights: {self.reward_weights}")

        self.X_val = x_val
        self.y_val = y_val
        self.X = X
        self.y = y

        self.task_name = task_name
        self.target_classes = num_classes

        # Load parameters
        self.controller_sampling_epochs = CONTROLLER_SAMPLING_EPOCHS
        self.samples_per_controller_epoch = SAMPLES_PER_CONTROLLER_EPOCH
        self.controller_train_epochs = CONTROLLER_TRAINING_EPOCHS
        self.architecture_train_epochs = ARCHITECTURE_TRAINING_EPOCHS
        self.controller_loss_alpha = CONTROLLER_LOSS_ALPHA

        # Accumulator of NAS data logs from training of sampled architectures
        # Note: (Sampled sequence, validation accuracy) with, optionally, predictor accuracy
        self.data = []

        # Path to save NAS data log to
        self.nas_data_log = 'LOGS/nas_data.pkl'

        # Clean LOGS folder from files
        clean_log()
        super().__init__(num_classes)

        self.model_generator = MLPGenerator(num_classes)

        # Limit the batch size to how many samples are generated per-epoch, with a maximum set to 32. Different approaches have been tried, from small to no batch size at all, but this has shown to give a good a balance between speed, memory's consumption and accuracy.
        self.controller_batch_size = SAMPLES_PER_CONTROLLER_EPOCH \
                if SAMPLES_PER_CONTROLLER_EPOCH < 32 else 32 # 32 as maximum batch size

        self.controller_input_shape = (1, MAX_ARCHITECTURE_LENGTH - 1) # Compliant with Keras input requirements

        if self.use_predictor:
            # Hybrid LSTM with surrogate accuracy predictor
            self.controller_model = self.hybrid_controller_model(self.controller_input_shape,
                                                                 self.controller_batch_size)
        else:
            # Simple one-layer LSTM
            self.controller_model = self.controller_model(self.controller_input_shape,
                                                          self.controller_batch_size)

    def create_architecture(self, sequence):
        """
        Create and compile a Keras model corresponding to the sequence.

        Returns
        -------
        model : keras.Model
            The compiled Keras architecture
        """

        if self.target_classes == 2:
            self.model_generator.mlp_loss_func = 'binary_crossentropy'

        model = self.model_generator.create_model(sequence, np.shape(self.X[0]))
        return self.model_generator.compile_model(model)

    def train_architecture(self, model):
        """
        Train a given Keras sampled architecture.

        Returns
        -------
        _ : keras.History
            Training history of the architecture.
            (Display purposes)
        """

        es_callback = EarlyStopping(monitor="val_loss", mode="min", patience=3, # Increase patience if more than 10 epochs
                                    verbose=1, restore_best_weights=True)

        # Return a permutation of the data
        X, y = unison_shuffled_copies(self.X, self.y)

        return self.model_generator.train_model(model, X, y,
                            self.architecture_train_epochs, callbacks=[es_callback])


    def append_model_metrics(self, sequence, history, pred_accuracy=None):
        """
        Append a new entry to the NAS log accumulator
        
        Parameters
        ----------
        sequence : list
            List of sampled IDs corresponding to entries in the vocabulary
            
        history : keras.History
            History object from training

        pred_accuracy : float [None]
            Predictor's accuracy on the sequence
        """

        if len(history.history['val_accuracy']) == 1: # Only one validation accuracy stored
            if pred_accuracy:
                self.data.append([sequence,
                                  history.history['val_accuracy'][0],
                                  pred_accuracy])
            else:
                self.data.append([sequence,
                                  history.history['val_accuracy'][0]])
            print('Validation accuracy: ', history.history['val_accuracy'][0])
        else: # Average out validation accuracies
            val_acc = np.ma.average(history.history['val_accuracy'], weights=np.arange(1,
                                    len(history.history['val_accuracy']) + 1), axis=-1)

            if pred_accuracy:
                self.data.append([sequence, val_acc,
                                  pred_accuracy])
            else:
                self.data.append([sequence, val_acc])
            print('Validation accuracy: ', val_acc)

    def prepare_controller_data(self, sequences):
        """
        Pad sequences to make them all of equal length as the input of the next controller's LSTM training. Take the last entry of the sequence as the target for the LSTM to predict given all the preceding entries.

        Returns
        -------
        val_acc_target : list
            List of validation accuracies as a target for the predictor
        """

        controller_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')

        # Reshape sequences accordingly to Keras's LSTM input and take them for t = 1, ..., T - 1
        Xc = controller_sequences[:, :-1].reshape(len(controller_sequences),
                                                  1, self.max_len - 1)

        # Categorize the last sequence's ID w.r.t. to the vocabulary W (self.controller_classes) as their value in [0, |W|) doesn't have to fool the controller in thinking that a higher value means something, as their all simply indexes.
        yc = to_categorical(controller_sequences[:, -1], self.controller_classes)
        val_acc_target = [item[1] for item in self.data]

        return Xc, yc, val_acc_target

    def get_discounted_reward(self, rewards):
        """
        Computes the discounted cumlative reward, whose expectation, we recall, is the objective that an agent (MLPNAS controller, here) in reinforcemnt learning tries to maximize through gradient descent of the policy gradient. It represents a way to evaluate a proxy of the total reinforcement received during the entire sequence of actions, starting from timestep t, that is, basically, the reward of the action that drives the environment frome the state s_t in the terminal state s_T, as a weighted sum of all rewards afterwards, with the underlying idea that far away rewards are exponentially less relevant.

        G(t) = sum_{k = t}^{T} gamma^{k - t}*r_k

        where gamma is the discount factor in [0, 1] to be applied to each future reward (from this, the adjective discounted). If gamma is 0 or close to it, then the agent only cares about the most immediate reward, whereas the higher it is the more it looks into the future, up to gamma equal to 1 when there is no discount at all.

        Parameters
        ----------
        rewards : list
            Reward values r_t for t = 1, ..., T, see custom_loss() for the details.
        """

        # Initiliaze cumulative reward to 0
        discounted_r = np.zeros_like(rewards, dtype=np.float32)

        for t in range(len(rewards)):
            running_add = 0. # Rewards sum accumulator
            exp = 0. # Discount factor exponent

            for r in rewards[t:]:
                running_add += self.controller_loss_alpha**exp * r
                exp += 1

            discounted_r[t] = running_add

        # The discounted reward is normalized (Z-score, here) for stability purposes, since its value affects the backpropagation equations, and in turn, it affects the gradients. By doing so, we keep its values in a specific convenient range, and we are also, in some way, encouraging and discouraging roughly half of the performed actions to the agent. http://karpathy.github.io/2016/05/31/rl/ 
        return (discounted_r - discounted_r.mean()) \
                        / discounted_r.std()

    def custom_loss(self, _, output):
        """
        Policy gradient: REINFORCE

        Parameters
        ----------
        _ : np.ndarray
            Targets array, which do not exist here.
            Note: Needed to make it compliant with a Keras loss.
        """
        
        # Compute the rewards by extracting the validation accuracies corresponding to the current epoch's sampled architectures and later soft threshold them w.r.t. the baseline value, of 0.5.
        baseline = 0.5
        reward = np.array([item[1] - baseline for item in self.data[-self.samples_per_controller_epoch:]]) \
                   .reshape(self.samples_per_controller_epoch, 1) # item[1] := Validation accuracy

        # Compute the discounted reward
        discounted_reward = self.get_discounted_reward(reward)

        # Compute REINFORCE loss. The negative sign is implied in the formula below, as we want to turn a minimization on the loss into a maximization problem, and this just makes it an equivalent problem.
        return -tf.math.log(output) * discounted_reward[:, None]

    def train_controller(self, model, X, y, pred_accuracy=None):
        """
        Train the controller splitting whether or not it is implying an hybrid LSTM with accuracy predictor.
        """
        if self.use_predictor:
            self.train_hybrid_model(model,
                                    X,
                                    y,
                                    pred_accuracy,
                                    self.custom_loss,
                                    self.controller_batch_size,
                                    self.controller_train_epochs)
        else:
            self.train_controller_model(model,
                                     X,
                                     y,
                                     self.custom_loss,
                                     self.controller_batch_size,
                                     self.controller_train_epochs)

    def search(self):
        """
        Navigate the search space for the desired number of epochs, looking for plausible architectures. The outer loop is related to the operations performed by the controller, whereas the inner loop is delegated to the MLP generator's operations.

        Returns
        -------
        self.data : list
            List of NAS data logs of the sampled architectures.
        """
        # For the number of controller epochs - Controller
        for controller_epoch in range(self.controller_sampling_epochs):
            print('------------------------------------------------------------------')
            print('                       Controller epoch: {}'.format(controller_epoch))
            print('------------------------------------------------------------------')
            # Sample a set number of architecture sequences
            sequences = self.sample_architecture_sequences(self.controller_model,
                                                           self.samples_per_controller_epoch)

            if self.use_predictor: # Predict their accuracies using a hybrid controller
                pred_accuracies = self.get_predicted_accuracies_hybrid_model(
                                                self.controller_model, sequences)

            # For each sampled sequence - MLP generator
            for i, sequence in enumerate(sequences):
                print('Architecture: ', self.decode_sequence(sequence))
                # Create, compile and train the corresponding model
                model   = self.create_architecture(sequence)
                history = self.train_architecture(model)

                # Log training metrics (w/ or w/o predictor)
                if self.use_predictor:
                    self.append_model_metrics(sequence, history, pred_accuracies[i])
                else:
                    self.append_model_metrics(sequence, history)
                print('------------------------------------------------------')

            # Sampled sequences are used data to train the controller
            Xc, yc, val_acc_target = self.prepare_controller_data(sequences)
            self.train_controller(self.controller_model, Xc, yc,
                                  val_acc_target[-self.samples_per_controller_epoch:]
            )

        # Log NAS data and event when search is over
        with open(self.nas_data_log, 'wb') as f:
            pickle.dump(self.data, f)
    
        self.model_generator.log_event()
        
        self.save_model_performance()
        return self.data

    def extract_best_model(self, search_data):
        """
        Extract the best architecture from the most recent NAS data and return it compiled.
        """
        search_data = sort_search_data(search_data)
        best_arch   = search_data[0][0] # [0][0] as it sorted
        
        return self.create_architecture(best_arch)

    def finetune_model(self, model, X, y, validation_split=0.2, 
                       batch_size=64, shuffle=True, epochs=50, save=True):
        """
        Fine-tune the model for another cycle of epochs on the training set (X, y).

        Parameters
        ----------
        save : bool [True]
            Whether to save a model checkpoint afterwards
        """      
        
        es_callback = EarlyStopping(monitor="val_loss", mode="min", patience=6,
                                    verbose=1, restore_best_weights=True)

        callbacks = [
            es_callback
        ]

        if save:
            mcp_save = ModelCheckpoint(os.path.join(BEST_MODEL_PATH, self.task_name + ".keras"),
                                       save_best_only=True, monitor='val_loss', mode='min')
            callbacks.append(mcp_save)
        
        history = model.fit(X, y, validation_split=validation_split, 
                            batch_size=batch_size, shuffle=True, 
                            epochs=epochs, callbacks=callbacks
        ) 

        print(model.summary())
        return history       

    @staticmethod
    def score(model, X_test, y_test):
        
        y_pred = model.predict(X_test)

        if len(y_pred.shape) > 1:
            y_pred_classes = y_pred.argmax(axis=-1)
        else:
            y_pred_classes = (y_pred > 0.5).astype(int) 

        f1 = f1_score(y_test, y_pred_classes, average='weighted') 

        auc = roc_auc_score(y_test, y_pred)

        test_loss, test_acc = model.evaluate(X_test, y_test)

        return (test_loss, test_acc, f1, auc)
    
    @staticmethod
    def shap_hot(model):
        import shap
        import pandas as pd
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        import os
        from sklearn.preprocessing import StandardScaler

        x_path = "mlp-nas/code/DATASETS/source_X.csv"
        y_path = "mlp-nas/code/DATASETSsource_y.csv"
        save_path = "mlp-nas/code/BARPLOTS/SHAP"
        os.makedirs(save_path, exist_ok=True)
        
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        target_name = y.columns[0]
        y = y.iloc[:, 0]

        n_per_class = min(300, min(sum(y == 0), sum(y == 1)))
        sampled_indices = (
            y[y == 0].sample(n=n_per_class, random_state=42).index.tolist() +
            y[y == 1].sample(n=n_per_class, random_state=42).index.tolist()
        )
        X_sampled = X.loc[sampled_indices]
        y_sampled = y.loc[sampled_indices]

        import re
        from collections import defaultdict

        def auto_group_features(feature_list):
            group_dict = defaultdict(list)
            for feat in feature_list:
                match = re.match(r"^(.*)_(\d+)$", feat)
                if match:
                    raw_group = match.group(1)
                else:
                    raw_group = feat
                split_parts = raw_group.split('_')
                group_name = '_'.join(split_parts[:2]) if len(split_parts) >= 2 else split_parts[0]
                group_name = group_name.title()
                group_dict[group_name].append(feat)
            return dict(group_dict)
        
        feature_groups = auto_group_features(X.columns.tolist())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sampled)

        background = shap.kmeans(X_scaled, 300)

        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_scaled[:1000], nsamples=200)
        shap_matrix = shap_values[0] if isinstance(shap_values, list) else shap_values
        
        def aggregate_shap(shap_matrix, feature_names, feature_groups):
            shap_matrix = np.squeeze(shap_matrix)
            shap_df = pd.DataFrame(shap_matrix, columns=feature_names)
            group_shap = {}
            for group, cols in feature_groups.items():
                existing = [c for c in cols if c in shap_df.columns]
                if existing:
                    group_shap[group] = shap_df[existing].sum(axis=1)
            return pd.DataFrame(group_shap)

        df_grouped = aggregate_shap(shap_matrix, X.columns.tolist(), feature_groups)

        level_idx = 4 
        quantile_values = {}
        for feature in df_grouped.columns:
            feature_shap = df_grouped[feature].values
            q = np.percentile(feature_shap, [0, 20, 40, 60, 80, 100])
            mask = (feature_shap >= q[level_idx]) & (feature_shap <= q[level_idx+1])
            if np.any(mask):
                quantile_values[feature] = np.mean(np.abs(feature_shap[mask]))
            else:
                quantile_values[feature] = 0
        top6_features = [x[0] for x in sorted(quantile_values.items(), key=lambda x: x[1], reverse=True)[:6]]

        heatmap_data = []
        for feature in top6_features:
            row = []
            feature_shap = df_grouped[feature].values
            q = np.percentile(feature_shap, [0, 20, 40, 60, 80, 100])
            for j in range(5):
                if j == 4:
                    mask = (feature_shap >= q[j]) & (feature_shap <= q[j+1])
                else:
                    mask = (feature_shap >= q[j]) & (feature_shap < q[j+1])
                if np.any(mask):
                    row.append(np.mean(np.abs(feature_shap[mask])))
                else:
                    row.append(0)
            heatmap_data.append(row)

        heatmap_df = pd.DataFrame(heatmap_data, index=top6_features, columns=[f'level-{i+1}' for i in range(5)])
        
        heatmap_df.index = [label.replace('_', '\n') for label in heatmap_df.index]
        
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".5f",
            cmap="Greens",
            linewidths=1,
            cbar=True,
            cbar_kws={'label': ''},
            annot_kws={"size": 18}  
        )
        
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=90,         
            ha='center',         
            rotation_mode='anchor', 
            fontsize=14, 
            weight='bold'
        )
        ax.set_xticklabels(                                                              
            ax.get_xticklabels(),
            fontsize=14,  
            weight='bold'
        )
        ax.tick_params(axis='y', pad=20) 
        ax.tick_params(axis='x', pad=15)
        
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)

        plt.tight_layout()
        pic_name = target_name + ".png"
        level_heatmap_path = os.path.join(save_path, pic_name)
        plt.savefig(level_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Level-5: {level_heatmap_path}")


