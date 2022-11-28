import os
import time
import datetime
import numpy as np
import math
import pandas as pd
from prepare import prep_env
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# Turn off the SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)

class Scaler(object):
    """
    Desc: Normalization utilities
    """
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        # type: (torch.tensor) -> None
        """
        Desc:
            Find mean and std with truncated columns
        Args:
            data:
        Returns:
            None
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        print("wind turbine line35: the shape of mean and std is {}, {}".format(self.mean.shape, self.std.shape))

    def transform(self, data):
        # type: (torch.tensor) -> torch.tensor
        """
        Desc:
            Transform the data
        Args:
            data:
        Returns:
            The transformed data
        """
        # data[:, 2:] -= mean
        # data[:, 2:] /= std
        # ****************** mean and std to(device) **************
        start_time = time.time()
        mean = torch.tensor(self.mean) if torch.is_tensor(data) else self.mean
        std = torch.tensor(self.std) if torch.is_tensor(data) else self.std

        data_no_time = data[:, 2:]
        # data_no_time = data[:, 1:]
        # print("wind turbine line51 \n", data_no_time[:5])
        data_no_time = (data_no_time - mean) / std
        data_std = np.hstack((data[:, :2], data_no_time))
        end_time = time.time()
        # print("wind turbine data line59: Elapsed time for standardizing data is {} secs \n".format(end_time - start_time))
        return data_std

    def inverse_transform(self, data):
        # type: (torch.tensor) -> torch.tensor
        """
        Desc:
            Restore to the original data
        Args:
            data: the transformed data
        Returns:
            The original data
        """
        # ****************** mean and std to(device) ************** 
        mean = torch.tensor(self.mean) if torch.is_tensor(data) else self.mean
        std = torch.tensor(self.std) if torch.is_tensor(data) else self.std

        data = data * std[-1] + mean[-1]

        return data


def time2obj(time_sj):
    data_sj = time.strptime(time_sj, "%H:%M")
    return data_sj


def time2int(time_sj):
    data_sj = time.strptime(time_sj, "%H:%M")
    time_int = int(time.mktime(data_sj))
    return time_int


def int2time(t):
    timestamp = datetime.datetime.fromtimestamp(t)
    return timestamp.strftime('"%H:%M"')


def func_add_t(x):
    time_strip = 600
    time_obj = time2obj(x)
    time_e = ((
        (time_obj.tm_sec + time_obj.tm_min * 60 + time_obj.tm_hour * 3600)) //
              time_strip) % 288
    return time_e


def func_add_h(x):
    time_obj = time2obj(x)
    hour_e = time_obj.tm_hour
    return hour_e

class BaseWPFDataset(Dataset):
    """
    Desc: Wind turbine power generation data
          Here, e.g.    15 days for training,
                        3 days for validation
    """
    first_initialized = False
    scaler_collection = None

    def __init__(
            self,
            data_path,
            mean_path,
            std_path,
            filename='wtbdata_245days.csv',
            new_filename='wtbdata_259days.csv',
            flag='train',
            use_new_data=False,
            task='MS',
            target='Patv',
            turbine_id=0,
            size=[36, 288],
            step_size=1, # actual input length is input_len / step_size
            capacity=134,
            day_len=24 * 6,
            scale=True,
            scale_cols=["Wspd", "Wdir", "Etmp", "Patv"],
            columns=["Day", "Tmstamp", "Wspd", "Wdir", "Etmp", "Patv"],
            start_col=1, # subject to change
            prev_train_days=214,  # 228 days
            new_train_days=228, 
            prev_val_days=31,  # 31 days
            new_val_days=31,
            prev_total_days=245,
            new_total_days=259,
            theta=0.9,
            is_test=False):

        super().__init__()
        self.unit_size = day_len
        self.input_len = size[0]
        self.output_len = size[1]
        self.step_size = step_size
        self.actual_input_len = self.input_len / self.step_size
        assert flag in ['train', 'val']
        self.flag = flag
        self.use_new_data = use_new_data
        type_map = {'train': 0, 'val': 1}
        self.set_type = type_map[flag]
        self.task = task
        self.target = target
        self.scale = scale
        self.scale_cols = scale_cols
        self.start_col = start_col
        self.feature_name = columns
        self.data_path = data_path
        self.mean_path = mean_path
        self.std_path = std_path
        self.tid = turbine_id
        self.capacity = capacity
        self.is_test = is_test
    
        if self.use_new_data and not self.is_test:
            self.filename = new_filename
            self.total_days = new_total_days
            self.train_days = new_train_days
            self.val_days = new_val_days
        
        else:
            self.filename = filename
            self.total_days = prev_total_days
            self.train_days = prev_train_days
            self.val_days = prev_val_days

        self.total_size = self.unit_size * self.total_days
        self.train_size = self.unit_size * self.train_days
        self.val_size = self.unit_size * self.val_days 

        if self.is_test:
            if not BaseWPFDataset.first_initialized:
                self.__read_data__()
                BaseWPFDataset.first_initialized = True
        else:
            self.__read_data__()

        if not self.is_test:
            self.data_x, self.data_y = self.__get_data__(self.tid)
        
    def __read_data__(self):
        self.df_raw = pd.read_csv(os.path.join(self.data_path, self.filename))
        print("wind turbine 200: <<<<<<the original dataset is>>>>>\n", self.df_raw[:5])
        if self.use_new_data and not self.is_test:
            self.df_raw = self.df_raw.drop(self.df_raw.columns[0], axis=1)
        # linear interpolation
        self.df_raw.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
        self.df_raw.interpolate(method='linear', limit_direction='backward', axis=0, inplace=True)
        self.df_raw.replace(to_replace=np.nan, value=0, inplace=True)
        # print("wind turnbine 190 \n", self.df_raw[:10])

    def __get_data__(self, turbine_id):
        data_x = self.__get_turbine__(turbine_id)
        data_y = data_x
        return data_x, data_y

    def __get_turbine__(self, turbine_id):
        border1s = [turbine_id * self.total_size,
                    turbine_id * self.total_size + self.train_size
                    ]
        border2s = [turbine_id * self.total_size + self.train_size,
                    turbine_id * self.total_size + self.train_size + self.val_size
                    ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # cols = self.df_raw.columns[self.start_col:]
        df_data = self.df_raw[self.feature_name]
        t = df_data['Tmstamp'].apply(func_add_t)
        df_data.insert(1, 'time', t)
        df_data = df_data.drop(columns='Tmstamp')
        start_time = time.time()
        df_data['time'] = df_data['time'].apply(lambda x: math.cos(2.0*math.pi*x / 144))
        df_data['Day'] =  df_data['Day'].apply(lambda x: math.cos(2.0*math.pi*x / 365))
        end_processing_time = time.time()
        print("wind turbine line234: Elapsed time for processing time stamp {}".format(end_processing_time - start_time))
        # train_data is for scaling use
        train_data = df_data[border1s[0]:border2s[0]][self.scale_cols]

        if not os.path.exists(self.mean_path) and not os.path.exists(self.std_path):
            os.mkdir(self.mean_path)
            os.mkdir(self.std_path)

        path_to_mean = os.path.join(self.mean_path, "mean{}.pt".format(turbine_id))
        path_to_std = os.path.join(self.std_path, "std{}.pt".format(turbine_id))
        
        if not os.path.exists(path_to_mean) and not os.path.exists(path_to_std):
            self.mean, self.std = self.__fit_and_save__(torch.Tensor(train_data.values), turbine_id)
        else:
            self.mean = torch.load(path_to_mean)
            self.std = torch.load(path_to_std)

        res_data = df_data[border1:border2]

        if self.scale:
            result = self.__transform__(res_data[self.scale_cols].values)
            if 'Tmstamp' in self.feature_name:
                print("lalalalalal")
                data_time = res_data['time'].values.reshape(-1, 1)
                result = np.hstack((data_time, result))
                print("260", result[:5])
            if 'Day' in self.feature_name:
                print("hahahahahahah")
                data_Day = res_data['Day'].values.reshape(-1, 1)
                result = np.hstack((data_Day, result))
                print("265", result[:5])
        else:
            result = res_data.values

        print("\n wind turbine line248 data after normalization: \n", result[:5])
        return result
    
    def __fit_and_save__(self, data, turbine_id):
        print("\nwind turbine line266: fitting data and calculating mean&std for turbine{}".format(turbine_id))
        mean = torch.mean(data, axis=0)
        std = torch.std(data, axis=0)
        path_to_mean = os.path.join(self.mean_path, "mean{}.pt".format(turbine_id))
        path_to_std = os.path.join(self.std_path, "std{}.pt".format(turbine_id))
        torch.save(mean, path_to_mean)
        torch.save(std, path_to_std)
        return mean, std

    def __transform__(self, data):

        mean = self.mean.numpy() if not torch.is_tensor(data) else self.mean
        std = self.std.numpy() if not torch.is_tensor(data) else self.std

        return (data - mean) / std
        
    def __getitem__(self, index):
        #
        # Sliding window with the size of input_len + output_len
        # the actual length of input is input_len / step_size
        # print("wind turbine new line252:", self.data_x.shape)
        if self.flag == "train" and self.use_new_data:
            boundary = 214 * self.unit_size - self.input_len - self.output_len + 1
            if (index < boundary):
                s_begin = index
                s_end = s_begin + self.input_len
                r_begin = s_end
                r_end = r_begin + self.output_len

                seq_x = self.data_x[s_begin:s_end][::self.step_size]
                seq_y = self.data_y[r_begin:r_end]
            else:
                s_begin = index + self.input_len + self.output_len - 1
                s_end = s_begin + self.input_len
                r_begin = s_end
                r_end = r_begin + self.output_len

                seq_x = self.data_x[s_begin:s_end][::self.step_size]
                seq_y = self.data_y[r_begin:r_end]
        
        else:
            s_begin = index
            s_end = s_begin + self.input_len
            r_begin = s_end
            r_end = r_begin + self.output_len

            seq_x = self.data_x[s_begin:s_end][::self.step_size]
            seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        # The number of samples of sliding windows method is calculated as follows
        if self.set_type < 2 and self.use_new_data:
            return int(len(self.data_x) - 2 * self.input_len - 2 * self.output_len + 2)
        # Otherwise,
        return int((len(self.data_x) - self.input_len - self.output_len + 1))

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

if __name__ == '__main__':
    envs = prep_env()
    data_set = BaseWPFDataset(
            data_path=envs["data_path"],
            filename=envs["filename"],
            flag='train',
            size=[envs["input_len"], envs["output_len"]],
            step_size=envs["step_size"],
            task=envs["task"],
            target=envs["target"],
            start_col=envs["start_col"],
            columns=envs["columns"],
            scale_cols=envs["scale_cols"],
            turbine_id=envs["turbine_id"],
            day_len=envs["day_len"],
            train_days=envs["train_days"],
            actual_train_days=envs["actual_train_days"],
            val_days=envs["val_days"],
            actual_val_days=envs["actual_val_days"],
            total_days=envs["total_days"],
            actual_total_days=envs["actual_total_days"]
        )
    print("wind turbine data \n", data_set.__getitem__(30816-144-288+1)[0])