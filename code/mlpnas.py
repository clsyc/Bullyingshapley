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
import keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers
# Keras utilities (v. 2.4.0)
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# Keras callbacks for training
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
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
    def __init__(self, X, y, num_classes, task_name='default'):
        """
        Parameters
        ----------
        task_name : str ['default']
            Name of the current task
            (Display purposes)
        """

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
        # return -K.log(output) * discounted_reward[:, None]

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
    def shapley(model):
        # 定义特征名称的映射
        columns_info = {
            'age_0': '11 years old or younger',
            'age_1': '12 years old',
            'age_2': '13 years old',
            'age_3': '14 years old',
            'age_4': '15 years old',
            'age_5': '16 years old',
            'age_6': '17 years old',
            'age_7': '18 years old or older',
            'sex_0': 'Female',
            'sex_1': 'Male',
            'Physically_attacked_0': 'Physically attacked 0 times',
            'Physically_attacked_1': 'Physically attacked 1 time',
            'Physically_attacked_2': 'Physically attacked 2 or 3 times',
            'Physically_attacked_3': 'Physically attacked 4 or 5 times',
            'Physically_attacked_4': 'Physically attacked 6 or 7 times',
            'Physically_attacked_5': 'Physically attacked 8 or 9 times',
            'Physically_attacked_6': 'Physically attacked 10 or 11 times',
            'Physically_attacked_7': 'Physically attacked 12 or more times',
            'Physical_fighting_0': 'Physical fighting 0 times',
            'Physical_fighting_1': 'Physical fighting 1 time',
            'Physical_fighting_2': 'Physical fighting 2 or 3 times',
            'Physical_fighting_3': 'Physical fighting 4 or 5 times',
            'Physical_fighting_4': 'Physical fighting 6 or 7 times',
            'Physical_fighting_5': 'Physical fighting 8 or 9 times',
            'Physical_fighting_6': 'Physical fighting 10 or 11 times',
            'Physical_fighting_7': 'Physical fighting 12 or more times',
            'Felt_lonely_0': 'Always fell lonely',
            'Felt_lonely_1': 'Never fell lonely',
            'Felt_lonely_2': 'Rarely fell lonely',
            'Felt_lonely_3': 'Sometimes fell lonely',
            'Felt_lonely_4': 'Most of the time fell lonely',
            'Close_friends_0': '0 close friends',
            'Close_friends_1': '1 close friends',
            'Close_friends_2': '2 close friends',
            'Close_friends_3': '3 or more close friends',
            'Miss_school_no_permission_0': 'Miss school no permission 0 days',
            'Miss_school_no_permission_1': 'Miss school no permission 1 to 2 days',
            'Miss_school_no_permission_2': 'Miss school no permission 3 to 5 days',
            'Miss_school_no_permission_3': 'Miss school no permission 6 to 9 days',
            'Miss_school_no_permission_4': 'Miss school no permission 10 or more days',
            'Other_students_kind_and_helpful_0': 'Other students never kind and helpful',
            'Other_students_kind_and_helpful_1': 'Other students sometimes kind and helpful',
            'Other_students_kind_and_helpful_2': 'Other students rarely kind and helpful',
            'Other_students_kind_and_helpful_3': 'Other students most of the time kind and helpful',
            'Other_students_kind_and_helpful_4': 'Other students always kind and helpful',
            'Parents_understand_problems_0': 'Parents always understand problems',
            'Parents_understand_problems_1': 'Parents never understand problems',
            'Parents_understand_problems_2': 'Parents most of the time understand problems',
            'Parents_understand_problems_3': 'Parents sometimes understand problems',
            'Parents_understand_problems_4': 'Parents rarely understand problems'
        }
        
        # 读取自变量和因变量
        X = pd.read_csv("/root/shapley_school/mlp-nas/code/DATASETS/schoolbullying/source_X.csv")
        y = pd.read_csv("/root/shapley_school/mlp-nas/code/DATASETS/schoolbullying/source_y1.csv")
        
        # 确保 y 是一个 Series
        y = y.iloc[:, 0]  # 假设 y 的目标变量在第一列
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 使用 K-Means 聚类将背景数据总结为  个聚类中心
        background_data = shap.kmeans(X_train, 100)
        explainer = shap.KernelExplainer(model, background_data)
        # 随机选择 1000 个测试样本
        X_test_sample = X_test.sample(n=1000, random_state=42)
        shap_values = explainer.shap_values(X_test_sample)

        # 定义需要合并的特征组
        feature_groups = {
            'age': [f'age_{i}' for i in range(8)],
            'sex': [f'sex_{i}' for i in range(2)],
            'Physically_attacked': [f'Physically_attacked_{i}' for i in range(8)],
            'Physical_fighting': [f'Physical_fighting_{i}' for i in range(8)],
            'Felt_lonely': [f'Felt_lonely_{i}' for i in range(5)],
            'Close_friends': [f'Close_friends_{i}' for i in range(4)],
            'Miss_school_no_permission': [f'Miss_school_no_permission_{i}' for i in range(5)],
            'Other_students_kind_and_helpful': [f'Other_students_kind_and_helpful_{i}' for i in range(5)],
            'Parents_understand_problems': [f'Parents_understand_problems_{i}' for i in range(5)]
        }

        # 创建一个新的特征矩阵和 SHAP 值矩阵
        new_X_test_sample = pd.DataFrame()
        new_shap_values = []

        for group_name, features in feature_groups.items():
            # 找到这些特征在 SHAP 值中的索引
            indices = [X.columns.get_loc(col) for col in features]
            # 计算这些特征的 SHAP 值
            
            print("shap_values[0].shape =", shap_values[0].shape)
            print("indices =", indices)

            
            group_shap_values = shap_values[0][:, indices]
            
            # 将 one-hot 编码的特征值转换为原始数值
            feature_values = np.dot(X_test_sample[features], np.arange(len(features)) + 1)
            
            # 将 SHAP 值求和并添加到新的特征矩阵中
            new_X_test_sample[group_name] = feature_values
            new_shap_values.append(np.sum(group_shap_values, axis=1))

        # 将新的 SHAP 值矩阵转换为 NumPy 数组
        new_shap_values = np.array(new_shap_values).T

        # 将合并后的 SHAP 值转换为 Explanation 对象
        shap_explanation = shap.Explanation(
            values=new_shap_values,
            base_values=explainer.expected_value[0],
            data=new_X_test_sample.values,
            feature_names=new_X_test_sample.columns
        )

        shap.plots.heatmap(shap_explanation)
        plt.savefig("/root/shapley_school/mlp-nas/code/BARPLOTS/shap_heatmap_plot.png", format="png", dpi=300)
        plt.close()

        # 遍历所有特征组，绘制 SHAP Scatter Plot
        for feature_name in feature_groups.keys():
            # 绘制 SHAP Scatter Plot
            shap.plots.scatter(shap_explanation[:, feature_name], show=True)
            
            # 保存为 PNG 文件
            plt.savefig(f"/root/shapley_school/mlp-nas/code/BARPLOTS/shap_scatter_plot_{feature_name}.png", format="png", dpi=300)
            plt.close()

        # 绘制摘要图
        plt.figure(figsize=(12, 8))  # 设置图表大小
        shap.summary_plot(new_shap_values, new_X_test_sample, feature_names=new_X_test_sample.columns, plot_type="bar")
        # 保存为 PNG 文件
        plt.savefig("/root/shapley_school/mlp-nas/code/BARPLOTS/shap_grouped_summary_plot.png", format="png", dpi=300, bbox_inches='tight')
        plt.close()

        # 创建新的特征名称列表
        new_feature_names = [columns_info.get(col, col) for col in X_test_sample.columns]

        # # 绘制特征重要性条形图
        plt.figure(figsize=(12, 8))  # 设置图表大小
        shap.summary_plot(shap_values[0], X_test_sample, feature_names=new_feature_names, plot_type="bar")
        # 保存为 PNG 文件
        plt.savefig("/root/shapley_school/mlp-nas/code/BARPLOTS/shap_bar_plot.png", format="png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 可视化全局特征重要性
        plt.figure(figsize=(12, 8))  # 设置图表大小
        shap.summary_plot(shap_values[0], X_test_sample, feature_names=new_feature_names)
        # 保存为 PNG 文件
        plt.savefig("/root/shapley_school/mlp-nas/code/BARPLOTS/shap_summary_plot.png", format="png", dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def score(model, X_test, y_test):
        # 使用模型对 X_test 进行预测
        y_pred = model.predict(X_test)

        # 将预测结果转换为与真实标签 y_test 相同的形状（如果是分类问题，通常会有分类结果）
        if len(y_pred.shape) > 1:
            # 对于多分类，获取每个类别的最大概率索引
            y_pred_classes = y_pred.argmax(axis=-1)
        else:
            # 对于二分类，预测值为概率
            y_pred_classes = (y_pred > 0.5).astype(int)  # 阈值 0.5 用于二分类

        # 计算 F1 score
        f1 = f1_score(y_test, y_pred_classes, average='weighted')  # 使用加权平均来处理不平衡类别

        auc = roc_auc_score(y_test, y_pred)

        # 计算损失和准确率
        test_loss, test_acc = model.evaluate(X_test, y_test)

        return (test_loss, test_acc, f1, auc)